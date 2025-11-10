"""my_cereal_task_builder.py: TFDS builder for our custom VLA dataset.

This version automatically splits the data into train, validation, and test sets.
"""

#Generation complete. 942 / 1000 successful samples saved. Part 1
#Generation complete. 940 / 1000 successful samples saved. Part 2
# Generation complete. 938 / 1000 successful samples saved. Part 3

import tensorflow as tf
import os
import json
import numpy as np
import tensorflow_datasets as tfds
import cv2  # You will need: pip install opencv-python
import random # For shuffling

class MyCerealTask(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for the custom cereal box task."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release with 80/10/10 train/val/test split.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Defines the dataset info."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': {
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                        ),
                    },
                    # This is our "action chunk"
                    'action': tfds.features.Tensor(
                        shape=(3, 7),  # 3 waypoints [A1, A2, A3], 7-DOF
                        dtype=np.float32,
                    ),
                    'language_instruction': tfds.features.Text(),
                    # RLDS requires these fields to define the episode
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                }),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        
        # --- IMPORTANT ---
        # Change this to the absolute path where your *raw* episode folders are
        data_dir = "/home/sherry/milestone/robosuite/my_data"
        
        #episode_paths = sorted(tfds.core.gfile.glob(os.path.join(data_dir, 'episode_*')))
        episode_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, 'episode_*')))
        print(f"Found {len(episode_paths)} total episodes.")
        
        if len(episode_paths) == 0:
            raise FileNotFoundError(f"No episode folders found at {data_dir}")

        # --- NEW: Shuffle and Split the data ---
        # Shuffle the paths for a random split
        random.seed(42) # Use a fixed seed for reproducibility
        random.shuffle(episode_paths)

        # Define split ratios (80% train, 10% val, 10% test)
        total_episodes = len(episode_paths)
        train_end = int(total_episodes * 0.8)
        val_end = int(total_episodes * 0.9)

        train_paths = episode_paths[:train_end]
        val_paths = episode_paths[train_end:val_end]
        test_paths = episode_paths[val_end:]

        print(f"Splitting data into:")
        print(f"  Train: {len(train_paths)} episodes")
        print(f"  Validation: {len(val_paths)} episodes")
        print(f"  Test: {len(test_paths)} episodes")
        # --- END OF NEW BLOCK ---

        return {
            'train': self._generate_examples(train_paths),
            'val': self._generate_examples(val_paths),
            'test': self._generate_examples(test_paths),
        }

    def _generate_examples(self, episode_paths):
        """Yields examples for a single split."""
        for episode_dir in episode_paths:
            metadata_path = os.path.join(episode_dir, 'metadata.json')
            image_path = os.path.join(episode_dir, 'image_rgb.png')

            try:
                # 1. Load Image
                # We read with cv2 and convert BGR -> RGB
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}. Skipping.")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 2. Load Metadata
                #with tfds.core.gfile.GFile(metadata_path, 'r') as f:
                with tf.io.gfile.GFile(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                instruction = metadata['instruction']
                waypoints = metadata['waypoints']

                # 3. Create the (3, 7) action chunk of ABSOLUTE poses
                # These are the labels the model will learn to predict.
                action_chunk = np.array([
                    waypoints['A1_pregrasp'],
                    waypoints['A2_grasp'],
                    waypoints['A3_release']
                ], dtype=np.float32)

                # 4. Assemble the single-step episode
                single_step = {
                    'observation': {
                        'image': image,
                    },
                    'action': action_chunk,
                    'language_instruction': instruction,
                    'is_first': True,
                    'is_last': True,
                    'is_terminal': True,
                }
                
                # Yield the episode
                episode_key = os.path.basename(episode_dir)
                yield episode_key, {'steps': [single_step]}

            except Exception as e:
                print(f"Error processing {episode_dir}: {e}. Skipping.")
                continue
