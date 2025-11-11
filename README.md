# Milestone Report: Language-Conditioned Affordances for Grasping in a Cluttered Environment

# Data Generation and Processing Pipeline

This document provides the instructions to run the full data pipeline, from initial data generation to final processing into a TFDS dataset in RLDS format (compatible with most VLA backbones like OpenVLA/Octo, etc). Assuming Robosuite 1.5.1 with all the required dependencies install to run it. 

## 1. Environment Setup

All commands should be run from the project's root directory (e.g., `robosuite/`).

1.  **Create and Activate Virtual Environment:**

    ```bash
    python3 -m venv venv # note that robosuite 1.5 should be compatible with python 3.10
    source venv/bin/activate
    ```

2.  **Install Project-specific Dependencies:**

    ```bash
    pip install tensorflow-datasets tensorflow opencv-python apache-beam mlcroissant
    ```

3.  **Create `__init__.py`:**
    This empty file is required for the `tfds build` import system to find the local builder script.
    
    ```bash
    touch __init__.py
    ```

---

## 2. Full Pipeline Execution

Please run the following commands in order from the project root directory.

### Step 1: Generate the VLA Dataset

This step generates the main dataset used for fine-tuning VLA. This command will call the custom-made pick_place_clutter.py in /environments/manipulation to generate 20 trials (only save the successful ones)and the demo videos for the first 2 samples for visual debugging.
Recommend to run this command to enable the visualization of the affordance labels. 

```bash
    export MUJOCO_GL=egl
    
    python generate_vla_dataset_visualize.py \
        --output_dir ./my_data \
        --num_trials 20 \
        --num_videos 2
```
Output: This creates a ./my_data directory containing episode_* folders and the corrponsponding videos.

The following is the command for generalizing samples in batches without visualiation of the affordance ground truth labels. 

```bash
    export MUJOCO_GL=egl
    
    python generate_vla_dataset.py \
        --output_dir ./my_data \
        --num_trials 500 \
        --num_videos 10 \
        --start_index 0
```
Output: This creates a ./my_data directory containing episode_* folders and the corrponsponding videos.

### Step 2: Visualize and Verify the VLA Dataset

This script confirms the generated data is valid and saves visualizations of the affordance waypoints.

```bash
    python visualize_dataset_and_affordances.py \
        --dataset_dir ./my_data \
        --output_dir ./my_data_VISUAL \
        --num_samples 5
```
Output: This prints waypoint(affordances) data to the console and saves visualization images (e.g., vis_episode_00000.png) into the ./my_data_VISUAL directory.

### Step 3: Configure TFDS Builder

Important: Before building, we must tell the builder script where to find the raw data (the ./my_data folder).

Navigate to the directory where we told generate_vla_dataset.py to save our data. For example: 

Bash
```
cd /home/sherry/milestone/robosuite/my_data
(If you had used the default, you would type cd ~/my_data)
```
List the contents to see if the episode folders are there:

Bash
```
ls
```
You should see an output that looks like this:

bash
```
episode_00000  episode_00001  episode_00002  episode_00003 ...
```
If you see that list of episode_... folders, you are in the correct place! Now, get the full absolute path by using pwd (print working directory):

Bash
```
pwd
```
The terminal will print the full, absolute path, for example: 

Bash
```
/home/sherry/milestone/robosuite/my_data
```

Copy that path (e.g., /home/sherry/milestone/robosuite/my_data).

Paste it into my_task_builder.py to replace the placeholder.

Before:

bash
```
Change this to the absolute path where your *raw* episode folders are
        data_dir = "/home/sherry/milestone/robosuite/my_data" 
```

After:

bash
```
        # Change this to the absolute path where your *raw* episode folders are
        data_dir = "<Your/Absoluate/Path/to/raw/date/episodes>" # <-- Paste your path here
```

### Step 4: Process Dataset into TFDS Format

This final step converts the raw ./my_data episodes into a tfrecord dataset.

Please make sure there is no folder named "processed_dataset" in the root project dir. 

Run the TFDS build command. The PYTHONPATH variable is necessary for the builder to find my_cereal_task_builder.

Bash
```
PYTHONPATH=$PYTHONPATH:. tfds build --imports="my_cereal_task_builder" MyCerealTask --data_dir="./processed_dataset"
```

## 3. Verification
Upon successful completion, we will have a ./processed_dataset directory containing the my_cereal_task/1.0.0/ dataset in tfrecord format. The terminal output will confirm the dataset generation and show a split (e.g., 8 train, 1 validation, 1 test).

bash
```
Found 19 total episodes.
Splitting data into:
  Train: 15 episodes
  Validation: 2 episodes
  Test: 2 episodes
```

And the data structure in RLDS format:

bash
```
    homepage='https://www.tensorflow.org/datasets/catalog/my_cereal_task',
    data_dir='processed_dataset/my_cereal_task/1.0.0',
    file_format=tfrecord,
    download_size=Unknown size,
    dataset_size=1.62 MiB,
    features=FeaturesDict({
        'steps': Dataset({
            'action': Tensor(shape=(3, 7), dtype=float32),
            'is_first': Scalar(shape=(), dtype=bool),
            'is_last': Scalar(shape=(), dtype=bool),
            'is_terminal': Scalar(shape=(), dtype=bool),
            'language_instruction': Text(shape=(), dtype=string),
            'observation': FeaturesDict({
                'image': Image(shape=(256, 256, 3), dtype=uint8),
            }),
        }),
    }),
    supervised_keys=None,
    disable_shuffling=False,
    nondeterministic_order=False,
    splits={
        'test': <SplitInfo num_examples=2, num_shards=1>,
        'train': <SplitInfo num_examples=15, num_shards=1>,
        'val': <SplitInfo num_examples=2, num_shards=1>,
    },
```

### Optional: Generate and a Seed Demo in HDF5 

This step creates an initial `demo.hdf5` file. But with the pipeline that executes from Step 1 to Step 4, we actually do not need this step for fine-tuning a VLA backbone. 

Run the seed generation script.
    ```bash
    python3 generate_seed_demo_1.5.py
    ```
    *(Note: This script is assumed to save its output to a specific location, e.g., `~/mimicgen_datasets/cereal_clutter/demo.hdf5`)*


## Full Datasets Access

### Please find the full 2820 successful trajectiores and raw datasets, as well as the processed RLDS dataset splits (train/val/test: 80%/20%/20%) here : https://drive.google.com/drive/folders/1JwtTHxw201sce3XsHMGfBErBaIzwrHWp?usp=sharing
