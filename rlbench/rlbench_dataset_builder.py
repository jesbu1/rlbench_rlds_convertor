from typing import Iterator, Tuple, Any
import pickle

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import PIL.Image as Image

import rlbench

from scipy.spatial.transform import Rotation as R

CAM_NAME = "front_rgb"
IMAGE_SHAPE = (224, 224, 3)
DELTA_ACTION = True
TRAIN_PATH = "/home/jeszhang/data/colosseum_dataset"
VAL_PATH = TRAIN_PATH  # temp for now TODO fix


def load_image(episode_path, image_folder, i):
    # load a png using PIL
    image = Image.open(f"{episode_path}/{image_folder}/{i}.png")
    image = image.resize((224, 224))
    # convert to numpy array
    data = np.array(image, dtype=np.uint8)
    assert data.shape == IMAGE_SHAPE
    return data


def convert_rlbench_action_to_tf_action(action):
    # [x, y, z, quaternion_x, quaternion_y, quaternion_z, quaternion_w, gripper] -> [x, y, z, euler_x, euler_y, euler_z, gripper]
    actions_euler = R.from_quat(action[3:7]).to_euler("xyz")
    return np.concatenate([action[:3], actions_euler, action[7:]])


class RLBench(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load(
        #    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        # )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=IMAGE_SHAPE,
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Front camera RGB observation.",
                                    ),
                                    # "state": tfds.features.Tensor(
                                    #    shape=(10,),
                                    #    dtype=np.float32,
                                    #    doc="Robot state, consists of [7x robot joint angles, "
                                    #    "2x gripper position, 1x door opening angle].",
                                    # ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [x, y, z, euler_x, euler_y, euler_z, gripper]. Uses delta action.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            # "language_embedding": tfds.features.Tensor(
                            #    shape=(512,),
                            #    dtype=np.float32,
                            #    doc="Kona language embedding. "
                            #    "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            # ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(path=TRAIN_PATH),
            "val": self._generate_examples(path=VAL_PATH),
        }

    def _generate_examples(
        self, path, language_instruction
    ) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, language_instruction: str):
            image_folder = CAM_NAME

            # language_embedding = self._embed([step["language_instruction"]])[
            #    0
            # ].numpy()
            with open(episode_path + "/low_dim_obs.pkl", "rb") as f:
                demo = pickle.load(f)

            gripper_poses = np.array(
                [
                    convert_rlbench_action_to_tf_action(
                        demo._observations[i].gripper_pose
                    )
                    for i in range(len(demo._observations))
                ]
            )

            episode = []
            prev_action = gripper_poses[0]
            # - 1 offset because we're predicting the next action
            for i in range(len(gripper_poses) - 1):
                curr_action = gripper_poses[i + 1]
                delta_action = curr_action - prev_action
                episode.append(
                    {
                        "observation": {
                            "image": load_image(episode_path, image_folder, i),
                            # "wrist_image": step["wrist_image"],
                            # "state": step["state"],
                        },
                        "action": delta_action if DELTA_ACTION else curr_action,
                        "is_first": i == 0,
                        "reward": float(i == (len(gripper_poses) - 2)),
                        "is_last": i == (len(gripper_poses) - 2),
                        "is_terminal": i == (len(gripper_poses) - 2),
                        "language_instruction": language_instruction,
                        #'language_embedding': language_embedding,
                    }
                )
                prev_action = curr_action

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {"file_path": episode_path}}

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples by recursively finding all subfolders in path with the name variation*
        variations_paths = glob.glob("f{path}/*/variation*", recursive=True)
        print(f"Found {len(variations_paths)} episodes in {path}")

        # now for each variation* path we load the language descriptions in `variation_descriptions.pkl`
        # and add them to the example
        for variation in variations_paths:
            # load language descriptions
            with open(f"{variation}/variation_descriptions.pkl", "rb") as f:
                language_descriptions = pickle.load(f)
            # randomly sample a language variation for each episode
            for episode_path in glob.glob(f"{variation}/episodes/episode*"):
                this_episode_lang_description = np.random.choice(language_descriptions)
                # for smallish datasets, use single-thread parsing
                yield _parse_example(episode_path, this_episode_lang_description)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
