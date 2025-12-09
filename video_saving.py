import robosuite as suite
from robosuite.utils import camera_utils
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2

from task_envs import MP2Env

#adding clustering
from sklearn.cluster import DBSCAN

#adding plotting packages
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#adding command-line argument for easy to run extra credit pipeline
import argparse


#=====================================================================
# Experiment
#=====================================================================

class Experiment:

    def __init__(self):
        
        # initialize environment
        self.env = MP2Env(
            robots = "Panda",
            has_renderer = False, # <--- CHANGE THIS TO TRUE will crash the mac rendering, save into videos instead
            has_offscreen_renderer = True,
            use_camera_obs = True,
            camera_names = ["birdview"],
            camera_depths = True,
            render_camera = "frontview",
            controller_configs = suite.load_composite_controller_config(
                controller = '{}/controllers/basic_abs_pose.json'.format(os.path.dirname(os.path.abspath(__file__)))
            ),
            horizon = 1e10,
        )
        self.obs = self.env.reset()

        # read initial robot pose
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self.obs["robot0_eef_quat_site"].copy()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        
        # To store the text for the video overlay
        self.trial_info_text = ""

    def move_to_pose (
            self,
            target_pos: np.array, 
            target_quat: np.array, 
            count: int, 
            time_for_residual_movement: int = 10
        ) -> None:
        '''
        Moves the robot to the target pose in a straight line.
        
        Parameters:
            - target_pos:   numpy array of size (3,) that specifies the target robot xyz position
            - target_quat:  numpy array of size (4,) that specifies the target robot quaternion
            - count:        an integer specifying the number of desired simulation timesteps this movement will take
        '''
        
        # rotation interpolation
        rotations = R.from_quat([self.robot_quat, target_quat])
        key_times = [0, 1]
        slerp = Slerp(key_times, rotations)

        for i in range (1, count+1):
            next_target_pos = (target_pos - self.robot_pos) * i/count + self.robot_pos
            next_target_quat = slerp(float(i)/count).as_quat()
            
            action = np.concatenate([next_target_pos, R.from_quat(next_target_quat).as_rotvec(degrees=False), [0]])
            self.obs, _, _, _ = self.env.step(action)
            
            # Adding saving video
            self._video_frame("move_to_pose")
            
        # wait a bit for any potential residual movement to complete
        for i in range (time_for_residual_movement):
            action = np.concatenate([target_pos, R.from_quat(target_quat).as_rotvec(degrees=False), [0]])
            self.obs, _, _, _ = self.env.step(action)

        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self.obs["robot0_eef_quat_site"].copy()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)

    def move_gripper (self, gripper_action: int) -> None:
        '''
        Operates the robot gripper to open or close.
        
        Parameters:
            - gripper_action: an integer; 1 closes the gripper, -1 opens the gripper, 0 does nothing
        '''
       
        for _ in range (10):
            action = np.concatenate([self.robot_pos, self.robot_rotvec, [gripper_action]])
            self.obs, _, _, _ = self.env.step(action)
            
            # Adding saving video
            self._video_frame("gripper")

    def sim_wait (self, count):
        for _ in range (count):
            action = np.concatenate([self.robot_pos, self.robot_rotvec, [0]])
            self.obs, _, _, _ = self.env.step(action)
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self.obs["robot0_eef_quat_site"].copy()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
    
    #=====================================================================
    # Video Saving Helpers
    # Because the macOS fails to open a rendering window
    # Using the below video saving helpers to directly save the videos
    #=====================================================================
    def _video_frame(self, text=None):
        if not getattr(self, "_rec", None) or not self._rec["on"]:
            return

        H, W = self._rec["H"], self._rec["W"]
        cam = self._rec["camera"]

        # Render
        rgb = self.env.sim.render(camera_name=cam, height=H, width=W, depth=False)
        frame = cv2.flip(rgb, 0)

        # Ensure dtype + contiguity for VideoWriter
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        # Optional overlays
        if text:
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        if self.trial_info_text:
            y = H - 10
            cv2.putText(frame, self.trial_info_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, self.trial_info_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr = np.ascontiguousarray(bgr)

        # Track frames-written to catch “empty” files
        self._rec["frames"] += 1
        self._rec["writer"].write(bgr)
        
    def _video_start(self, path="Q4.mp4", fps=30, H=256, W=256, camera_name="birdview"):
        self._rec = {"on": False, "path": path, "fps": fps, "H": H, "W": W, "camera": camera_name, "frames": 0}
        for fourcc_str in ("avc1", "mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(path, fourcc, fps, (W, H), True)
            if writer.isOpened():
                self._rec.update({"writer": writer, "fourcc": fourcc_str, "on": True})
                break
        if not self._rec["on"]:
            raise RuntimeError(f"Failed to open VideoWriter for {path}. Install H.264 support or try .avi with XVID.")

        # Warm up the renderer once after previous resets
        _ = self.env.sim.render(camera_name=camera_name, height=H, width=W, depth=False)

        # Seed file with a few frames so it’s never near-empty
        for _ in range(3):
            self._video_frame("start")

    def _video_stop(self):
        if getattr(self, "_rec", None) and self._rec.get("on", False):
            self._rec["writer"].release()
            print(f"[VIDEO] Saved to {self._rec['path']} (codec={self._rec.get('fourcc')}, frames={self._rec['frames']})")
            self._rec["on"] = False

 
    #=====================================================================
    # Original Experiment Pipeline (Single Trial Functions)
    #=====================================================================
    #    For a single trial
    def run_original_pipeline (self, trial_num, total_trials):
        """
        Runs one trial of the original (Q1-Q5) pipeline.
        Returns:
            bool: True if successful, False otherwise.
        """
        # Set trial num for logging helpers
        self.trial_num = trial_num
        
        # Start video recording
        # Update the text overlay for the video
        self.trial_info_text = f"O-Trial: {trial_num}/{total_trials} | In Progress"
        self._video_start(f"Q5_O-trial_{trial_num}.mp4", fps=30, H=256, W=256, camera_name="birdview")
        print("[VIDEO] Recording trial {trial_num} to {video_path}...")
        
        # --- Constants ---
        RED_CUBE_HEIGHT = 0.02
        GREEN_CUBE_HEIGHT = 0.025
        # Standard top-down orientation for placing
        standard_place_quat = R.from_euler("xyz", [180., 0., 90.], degrees=True).as_quat()
        
        # --- 1. Perception ---
        camera_name = "birdview"
        H = 256
        W = 256
        intrinsics = camera_utils.get_camera_intrinsic_matrix(self.env.sim, "birdview", H, W)
        extrinsics = camera_utils.get_camera_extrinsic_matrix(self.env.sim, "birdview")
        
        # Q1: Get color masks
        display_image, red_mask, green_mask = self.Q1_pipeline()
        
        # Q2: Get world point cloud
        pts_w, is_red, is_green, v, u = self.Q2_pipeline(red_mask, green_mask, intrinsics, extrinsics)
        
        # Find cube centers
        cube_poses = self.Perception_Pipeline_original(pts_w, is_red, is_green)
        
        if cube_poses:
            # --- 2. Trajectory Planning & Execution ---
            
            # --- STAGE 1: Pick Red Cube ---
            print("\n--- STAGE 1: Picking Red Cube ---")

            cluttered_vec = cube_poses["green_2"] - cube_poses["green_1"]
            x_axis_gripper = np.array([-cluttered_vec[1], cluttered_vec[0], 0])
            x_axis_gripper /= np.linalg.norm(x_axis_gripper)
            z_axis_gripper = np.array([0., 0., -1.])
            y_axis_gripper = np.cross(z_axis_gripper, x_axis_gripper)
            grasp_rotation_matrix = np.array([x_axis_gripper, y_axis_gripper, z_axis_gripper]).T
            red_grasp_quat = R.from_matrix(grasp_rotation_matrix).as_quat()
            
            self.Pick_Cube(cube_poses["red"], red_grasp_quat)
            self.sim_wait(60) # Hold the cube

            # --- STAGE 2: Place Red Cube ---
            print("\n--- STAGE 2: Placing Red Cube ---")

            green_1_pos = cube_poses["green_1"]
            green_2_pos = cube_poses["green_2"]
            dist_to_green_1 = np.linalg.norm(self.robot_pos - green_1_pos)
            dist_to_green_2 = np.linalg.norm(self.robot_pos - green_2_pos)

            if dist_to_green_1 > dist_to_green_2:
                base_green_pos = green_1_pos
                second_green_pos = green_2_pos
            else:
                base_green_pos = green_2_pos
                second_green_pos = green_1_pos

            # Calculate place position
            place_pos_red = np.array([base_green_pos[0], base_green_pos[1],
                                      base_green_pos[2] + GREEN_CUBE_HEIGHT / 2 + RED_CUBE_HEIGHT / 2])
            
            self.Place_Cube(place_pos_red, standard_place_quat, is_final_place=False)

            # --- STAGE 3: Pick Second Green Cube ---
            print("\n--- STAGE 3: Picking Second Green Cube ---")
            self.Pick_Cube(second_green_pos, standard_place_quat)

            # --- STAGE 4: Place Top Green Cube ---
            print("\n--- STAGE 4: Placing Top Green Cube ---")
            # Calculate final place position
            place_pos_final = np.array([place_pos_red[0], place_pos_red[1],
                                        place_pos_red[2] + RED_CUBE_HEIGHT / 2 + GREEN_CUBE_HEIGHT])
            
            self.Place_Cube(place_pos_final, standard_place_quat, is_final_place=True)

            # --- STAGE 5: Move to Safe Position ---
            print("\n--- STAGE 5: Moving to Safe Position ---")
            self.move_to_pose(
                self.robot_pos - np.array([0.2, 0.2, 0]),
                R.from_euler("xyz", [180., 0., 90.], degrees=True).as_quat(),
                count=50
            ) #

            # --- 3. Evaluation ---
            print("\nWaiting for 3 seconds to check stack stability...")
            self.sim_wait(60) # 20Hz control_freq * 3 seconds = 60 steps
            for _ in range(60): # 3 seconds * 20 Hz = 60 steps
                action = np.concatenate([self.robot_pos, self.robot_rotvec, [0]])
                self.obs, _, _, _ = self.env.step(action)
                self._video_frame() # Record a frame on each step of the wait
            
            self.robot_pos = self.obs["robot0_eef_pos"].copy()
            self.robot_quat = self.obs["robot0_eef_quat_site"].copy()
            self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
            
            trial_successful = self.check_stack_success_original()
            if trial_successful:
                print("Trial SUCCEEDED.")
                self.trial_info_text = f"O-trial: {trial_num}/{total_trials} | Success"
            else:
                print("Trial FAILED based on final pose check.")
                self.trial_info_text = f"O-Trial: {trial_num}/{total_trials} | Failed (Stack)"
                
            # Record final result for a few seconds
            for _ in range(60):
                action = np.concatenate([self.robot_pos, self.robot_rotvec, [0]])
                self.obs, _, _, _ = self.env.step(action)
                self._video_frame()
                
            self._video_frame()
            self._video_stop()
            
            return trial_successful
                
        else:
            print("TRIAL FAILED: Could not identify cubes.")
            self.trial_info_text = f"O-Trial: {trial_num}/{total_trials} | Failed (Perception)"

            # Record final result for a few seconds
            for _ in range(60):
                action = np.concatenate([self.robot_pos, self.robot_rotvec, [0]])
                self.obs, _, _, _ = self.env.step(action)
                self._video_frame()
                
            self._video_frame()
            self._video_stop()
            
            return False
            
    #=====================================================================
    # Extra Credit Experiment Pipeline (Single Trial Functions)
    #=====================================================================
        
  
        
    def run_trials(self, extra_credit=False, total_trials=1):
        """
        Runs the full pipeline for a specified number of trials.
        """
        success_count = 0
        
        print(f"\n{'='*20} STARTING PIPELINE {'='*20}")
        print(f"Running {total_trials} trial(s)...")
        print(f"Extra Credit Mode: {extra_credit}")
        
        for trial_num in range(1, total_trials + 1):
            print(f"\n{'='*20} TRIAL {trial_num} / {total_trials} {'='*20}")
            
            trial_successful = False
            try:
                if extra_credit:
                    trial_successful = self.run_extra_pipeline(trial_num, total_trials)
                else:
                    trial_successful = self.run_original_pipeline(trial_num, total_trials)
            
            except Exception as e:
                print(f"TRIAL FAILED: An unexpected error occurred during trial {trial_num}: {e}")
                trial_successful = False # Ensure it's marked as failed

            if trial_successful:
                success_count += 1
                print(f"TRIAL {trial_num} SUCCEEDED.")
            else:
                print(f"TRIAL {trial_num} FAILED.")

            # Reset the environment for the next trial, if not the last trial
            if trial_num < total_trials:
                print("Resetting environment for the next trial...")
                self.obs = self.env.reset()
                self.robot_pos = self.obs["robot0_eef_pos"].copy()
                self.robot_quat = self.obs["robot0_eef_quat_site"].copy()
                self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)

        # --- Final Report ---
        success_rate = (success_count / total_trials) * 100
        print(f"\n{'='*20} FINAL RESULTS {'='*20}")
        print(f"Total Trials: {total_trials}")
        print(f"Successful Trials: {success_count}")
        print(f"Success Rate: {success_rate:.1f}%")

        # steps simulation but does not command the robot
        # while True:
        #     action = np.concatenate([self.robot_pos, self.robot_rotvec, [0]])
        #     self.env.step(action)

#=====================================================================
# Added arg to help run trials between original and extra credit pipelines
#=====================================================================

if __name__ == "__main__":
#    exp = Experiment()
#    exp.run()
    
    parser.add_argument(
        '--num_trials',
        type=int,
        default=1,
        help='Number of trials to run'
    )
    
    args = parser.parse_args()
    
    # --- Run experiment with the flag ---
    exp = Experiment()
    # I don't recommend ro run with --num_trials <more than 2> due to macOS flickering issue
    exp.run_trials(extra_credit=args.extra_credit, total_trials=args.num_trials)



