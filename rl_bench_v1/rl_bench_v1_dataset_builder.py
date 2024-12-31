from typing import Iterator, Tuple, Any
import pickle
import h5py

import glob
from rl_bench_v1.conversion_utils import MultiThreadedDatasetBuilder
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import PIL.Image as Image
import io

import rlbench

from scipy.spatial.transform import Rotation as R

CAM_NAME = "front_rgb"
IMAGE_SHAPE = (224, 224, 3)
DELTA_ACTION = True
TRAIN_PATH = "/home/liyi/workspace/dataset/rlbench_all_tasks_256_1000_eps_compressed/"
VAL_PATH = ""  # temp for now TODO fix
DEBUG = False
SKIP_VAL = VAL_PATH == ""
exclude_tasks = ["basketball_in_hoop", "change_channel",  "empty_dishwasher", "get_ice_from_fridge",
                         "open_oven", "plug_charger_in_power_supply", "put_books_on_bookshelf", "put_tray_in_oven",
                         "take_cup_out_from_cabinet", "take_tray_out_of_oven", "tv_on", "unplug_charger",
                         "move_hanger", "turn_oven_on", 'press_switch', 'close_fridge', 'hang_frame_on_hanger',
                         'open_fridge', 'put_books_at_shelf_location', "take_frame_off_hanger", "take_off_weighing_scales",
                         "take_shoes_out_of_box"]


def _generate_examples(path) -> Iterator[Tuple[str, Any]]:
    """Generator of examples for each split."""
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_example(
        episode_path,
        which_episode: int,
        total_episodes: int,
    ):
        with open(episode_path + "/variation_descriptions.pkl", "rb") as f:
            language_instructions = pickle.load(f)
        language_instruction = np.random.choice(language_instructions)
        language_embedding = _embed([language_instruction])[0].numpy()
        with open(episode_path + "/low_dim_obs.pkl", "rb") as f:
            demo = pickle.load(f)

        gripper_poses = np.array(
            [
                get_action_from_obs(demo._observations[i])
                for i in range(len(demo._observations))
            ]
        ).astype(np.float32)

        episode = []
        prev_action = gripper_poses[0]
        # - 1 offset because we're predicting the next action
        # load the images from an h5 file
        with h5py.File(episode_path + f"/{CAM_NAME}.h5", "r") as images:
            for i in range(len(gripper_poses) - 1):
                curr_action = gripper_poses[i + 1]
                delta_action = curr_action - prev_action
                episode.append(
                    {
                        "observation": {
                            "image": load_image(images, i),
                        },
                        "action": delta_action if DELTA_ACTION else curr_action,
                        "discount": 1.0,
                        "is_first": i == 0,
                        "reward": float(i == (len(gripper_poses) - 2)),
                        "is_last": i == (len(gripper_poses) - 2),
                        "is_terminal": i == (len(gripper_poses) - 2),
                        "language_instruction": language_instruction,
                        "language_embedding": language_embedding,
                    }
                )
                prev_action = curr_action

        # create output data sample
        sample = {"steps": episode, "episode_metadata": {"file_path": episode_path}}

        print(
            f"---------------------Generated {which_episode} episode out of {total_episodes}---------------------"
        )

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # create list of all examples by recursively finding all subfolders in path with the name variation*
    variations_paths = glob.glob(
        f"{path}/*/*variation*", recursive=True
    )  # Colosseum or RLBench
    # variations_paths = glob.glob(f"{path}/*/all_variations", recursive=True) # Just RLBench

    if DEBUG:
        print("---------------------DEBUG MODE-----------------------------")
        variations_paths = variations_paths[:2]

    # now for each variation* path we load the language descriptions in `variation_descriptions.pkl`
    # and add them to the example
    print(
        f"---------------------Found {len(variations_paths)} variations in {path}--------------------"
    )
    examples = []
    for variation in variations_paths:
        # load language descriptions

        for episode_path in glob.glob(f"{variation}/episodes/episode*"):
            if any([excluded_name in episode_path for excluded_name in exclude_tasks]):
                continue
            examples.append(
                (
                    episode_path,
                    len(examples),
                )
            )

    print(
        f"-----------------------Will create {len(examples)} trajectories from {path}--------------------------------"
    )
    # add len(examples) to each example so we can track progress
    examples = [(*example, len(examples)) for example in examples]

    for example in examples:
        yield _parse_example(*example)


def load_image(image_h5, i):
    # load a png using PIL
    image_string = image_h5[f"image_{i}"][()]
    image = Image.open(io.BytesIO(image_string))
    image = image.resize((224, 224))
    # convert to numpy array
    data = np.array(image, dtype=np.uint8)
    assert data.shape == IMAGE_SHAPE
    return data


def get_action_from_obs(obs):
    # [x, y, z, quaternion_x, quaternion_y, quaternion_z, quaternion_w, gripper] -> [x, y, z, euler_x, euler_y, euler_z, gripper]
    gripper_pose = obs.gripper_pose
    gripper_open = np.array([obs.gripper_open])
    gripper_close = 1 - gripper_open
    actions_euler = R.from_quat(gripper_pose[3:]).as_euler("xyz")
    return np.concatenate([gripper_pose[:3], actions_euler, gripper_close])


class RLBenchV1(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 5  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = (
        50  # number of paths converted & stored in memory before writing to disk
    )
    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = (
        _generate_examples  # handle to parse function from file paths to RLDS episodes
    )

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
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
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
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
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

    def _split_paths(self):
        """Define data splits."""
        ret_dict = {
            "train": self._generate_examples(path=TRAIN_PATH),
        }

        if DEBUG or SKIP_VAL:
            return ret_dict

        ret_dict.update({"val": self._generate_examples(path=VAL_PATH)})
        return ret_dict