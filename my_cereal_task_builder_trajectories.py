"""

TensorFlow Datasets (TFDS) Builder for the custom VLA Cereal Task
using FULL TRAJECTORIES (multi-step episodes).

This builder:

- Reads raw episodes from a directory that contains folders like:
      episode_00000/
          image_rgb.png        # (optional, not used by this builder)
          image_depth.npy      # (optional, not used by this builder)
          metadata.json
          trajectory.npz       # per-timestep data

- Expects `trajectory.npz` to contain:
      images   : [T, H, W, 3] uint8  (per-timestep RGB)
      actions  : [T, 7] float32      (x, y, z, rotvec(3), gripper)
      eef_pos  : [T, 3] float32
      eef_quat : [T, 4] float32
      gripper  : [T]   float32

- Expects `metadata.json` to contain at least:
      {
          "instruction": "Pick up the cereal box.",
          "affordance_waypoints": {...},         # optional
          "has_full_trajectory": true            # optional
      }

- Converts each episode into an RLDS-style structure with:
    * steps[t]:
        observation:
          image:      [H, W, 3] uint8
          eef_pos:    [3] float32
          eef_quat:   [4] float32
          gripper:    []  float32
        action:        [7] float32
        language_instruction: str (same for all steps in episode)
        reward:       float32 (1.0 only on final step)
        is_first:     bool
        is_last:      bool
        is_terminal:  bool

    * episode_metadata:
        instruction: str
        success:     bool
        waypoints:   JSON string of affordance_waypoints (optional)

- Splits episodes into train/val/test with 80/10/10 ratio using a fixed
  random seed (42) for reproducibility.
"""

import os
import json
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# ----------------------------------------------------------------------
# IMPORTANT: Set this to the folder that contains your episode_* dirs
# e.g. "/scratch/.../cereal_dataset" such that:
#       RAW_DATA_DIR/
#           episode_00000/
#           episode_00001/
#           ...
# ----------------------------------------------------------------------
RAW_DATA_DIR = "/projects/illinois/eng/cs/namato/xuehail2/gs_sh/498RM_Project_Report/test_output"  

class MyCerealTaskTrajectories(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for the cereal VLA task with full trajectories."""

    VERSION = tfds.core.Version("0.2.0")
    RELEASE_NOTES = {
        "0.2.0": "Full multi-step trajectories with per-timestep images.",
    }

    # Required if we want to safely access dl_manager.manual_dir
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "Place episode_* directories (with metadata.json and trajectory.npz) "
        "inside the manual_dir you pass to tfds.build."
    )

    def _info(self) -> tfds.core.DatasetInfo:
        """Defines the dataset schema."""

        # Change this if your env uses a different resolution.
        image_shape = (256, 256, 3)

        step_features = {
            "observation": tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(
                        shape=image_shape,
                        dtype=np.uint8,
                        encoding_format="png",
                    ),
                    "eef_pos": tfds.features.Tensor(
                        shape=(3,),
                        dtype=np.float32,
                    ),
                    "eef_quat": tfds.features.Tensor(
                        shape=(4,),
                        dtype=np.float32,
                    ),
                    "gripper": tfds.features.Tensor(
                        shape=(),
                        dtype=np.float32,
                    ),
                }
            ),
            "action": tfds.features.Tensor(
                shape=(7,),  # 7-DOF: [x,y,z, rotvec(3), gripper]
                dtype=np.float32,
            ),
            "language_instruction": tfds.features.Text(),
            "reward": tfds.features.Scalar(dtype=np.float32),
            "is_first": tfds.features.Scalar(dtype=np.bool_),
            "is_last": tfds.features.Scalar(dtype=np.bool_),
            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
        }

        episode_metadata_features = tfds.features.FeaturesDict(
            {
                "instruction": tfds.features.Text(),
                "success": tfds.features.Scalar(dtype=np.bool_),
                # Store waypoints as JSON string for flexibility
                "waypoints_json": tfds.features.Text(),
            }
        )

        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Cereal pick-and-place VLA dataset with full robot "
                "trajectories and per-timestep RGB observations."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(step_features),
                    "episode_metadata": episode_metadata_features,
                }
            ),
            homepage="https://example.com/my_cereal_task",  # optional / placeholder
        )

    def _split_generators(self, dl_manager):
        """Create train/val/test splits from the raw episode directories."""

        # Prefer manual_dir if user supplied one, else use RAW_DATA_DIR
        if dl_manager.manual_dir:
            data_dir = dl_manager.manual_dir
        else:
            data_dir = RAW_DATA_DIR

        if not os.path.isdir(data_dir):
            raise ValueError(
                f"Raw data directory '{data_dir}' does not exist. "
                "Set RAW_DATA_DIR in the builder or pass --manual_dir=... "
                "when calling `tfds build`."
            )

        # Collect all episode_* directories
        episode_dirs: List[str] = []
        for name in os.listdir(data_dir):
            full = os.path.join(data_dir, name)
            if os.path.isdir(full) and name.startswith("episode_"):
                episode_dirs.append(full)

        episode_dirs = sorted(episode_dirs)

        if not episode_dirs:
            raise ValueError(f"No 'episode_*' folders found in {data_dir}")

        # Shuffle with fixed seed for reproducible split
        rng = np.random.RandomState(42)
        rng.shuffle(episode_dirs)

        n_total = len(episode_dirs)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val

        train_dirs = episode_dirs[:n_train]
        val_dirs = episode_dirs[n_train : n_train + n_val]
        test_dirs = episode_dirs[n_train + n_val :]

        print(
            f"Found {n_total} episodes in {data_dir} -> "
            f"{len(train_dirs)} train, {len(val_dirs)} val, {len(test_dirs)} test."
        )

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"episode_dirs": train_dirs},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={"episode_dirs": val_dirs},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"episode_dirs": test_dirs},
            ),
        ]

    def _generate_examples(self, episode_dirs: List[str]):
        """Yields RLDS-style episodes from the list of episode directories."""

        for episode_dir in episode_dirs:
            episode_key = os.path.basename(episode_dir)

            try:
                # --------------------------------------------------------------
                # 1) Load metadata.json
                # --------------------------------------------------------------
                meta_path = os.path.join(episode_dir, "metadata.json")
                if not os.path.exists(meta_path):
                    print(f"[WARN] No metadata.json in {episode_dir}, skipping.")
                    continue

                with open(meta_path, "r") as f:
                    meta = json.load(f)

                instruction = meta.get("instruction", "")
                waypoints = meta.get("affordance_waypoints", {})
                success_flag = bool(meta.get("has_full_trajectory", True))
                waypoints_json = json.dumps(waypoints)

                # --------------------------------------------------------------
                # 2) Load trajectory.npz (per-timestep data)
                # --------------------------------------------------------------
                traj_path = os.path.join(episode_dir, "trajectory.npz")
                if not os.path.exists(traj_path):
                    print(f"[WARN] No trajectory.npz in {episode_dir}, skipping.")
                    continue

                traj = np.load(traj_path)

                images = traj["images"]      # [T, H, W, 3], uint8
                actions = traj["actions"]    # [T, 7], float32
                eef_pos = traj["eef_pos"]    # [T, 3], float32
                eef_quat = traj["eef_quat"]  # [T, 4], float32
                gripper = traj["gripper"]    # [T],   float32

                T = actions.shape[0]
                if (
                    images.shape[0] != T
                    or eef_pos.shape[0] != T
                    or eef_quat.shape[0] != T
                    or gripper.shape[0] != T
                ):
                    print(
                        f"[WARN] Mismatched trajectory lengths in {episode_dir}, "
                        f"skipping."
                    )
                    continue

                # --------------------------------------------------------------
                # 3) Build multi-step RLDS episode
                # --------------------------------------------------------------
                steps: List[Dict[str, Any]] = []

                for t in range(T):
                    obs = {
                        "image": images[t],  # uint8 [H, W, 3]
                        "eef_pos": eef_pos[t].astype(np.float32),
                        "eef_quat": eef_quat[t].astype(np.float32),
                        "gripper": np.float32(gripper[t]),
                    }

                    step = {
                        "observation": obs,
                        "action": actions[t].astype(np.float32),
                        "language_instruction": instruction,
                        "reward": np.float32(1.0 if t == T - 1 else 0.0),
                        "is_first": bool(t == 0),
                        "is_last": bool(t == T - 1),
                        "is_terminal": bool(t == T - 1),
                    }

                    steps.append(step)

                episode_metadata = {
                    "instruction": instruction,
                    "success": success_flag,
                    "waypoints_json": waypoints_json,
                }

                yield episode_key, {
                    "steps": steps,
                    "episode_metadata": episode_metadata,
                }

            except Exception as e:
                print(f"[ERROR] Failed to process {episode_dir}: {e}")
                continue
