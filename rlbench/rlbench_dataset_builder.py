from typing import Iterator, Tuple, Any
import pickle

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import rlbench

CAM_NAME = "front_rgb"


class RLBench(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
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
                                        shape=(224, 224, 3),
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
            "train": self._generate_examples(path="data/train/episode_*.npy"),
            "val": self._generate_examples(path="data/val/episode_*.npy"),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, language_instruction: str):
            image_folder = CAM_NAME
            # TODO: load `low_dim_obs.pkl` but it needs rlbench dependency

            # language_embedding = self._embed([step["language_instruction"]])[
            #    0
            # ].numpy()

            # TODO: use delta action

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                episode.append(
                    {
                        "observation": {
                            "image": step["image"],
                            # "wrist_image": step["wrist_image"],
                            "state": step["state"],
                        },
                        "action": step["action"],
                        "is_first": i == 0,
                        "is_last": i == (len(data) - 1),
                        "is_terminal": i == (len(data) - 1),
                        #'language_embedding': language_embedding,
                        "language_instruction": step["language_instruction"],
                    }
                )

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
                # add language descriptions to each episode
                for i, step in enumerate(np.load(episode_path, allow_pickle=True)):
                    step["language_instruction"] = language_descriptions[i]
                # for smallish datasets, use single-thread parsing
                yield _parse_example(episode_path)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
