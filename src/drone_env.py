"""
=====================================================
Drone Environment Assignment (Gymnasium + CoppeliaSim)
Course: Technologies and IoT Applications
University: Harokopio University of Athens (DIT)
Katsimpras Drosos - ais25123@hua.gr
=====================================================

Required components to implement:
- __init__()
- reset()
- step()
- _get_obs()

Required elements:
- Action space
- Observation space
- Reward function
- Termination logic
- CoppeliaSim connection
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import cv2

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class DroneEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Connect to CoppeliaSim
        # - Initialize the Remote API Client
        # - Access the 'sim' object
        # - Retrieve all required handles (drone, target, camera, obstacles)
 
        print("Connecting to CoppeliaSim...")
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # grab main object handles from the scene
        self.drone = self.sim.getObject('/Quadcopter')
        self.camera = self.sim.getObject('/Quadcopter/visionSensor')
        self.target = self.sim.getObject('/my_target')

        # put all walls and blocks in a list so it's easier to check for crashes later
        self.obstacles = [
            self.sim.getObject('/WALL1'),
            self.sim.getObject('/WALL2'),
            self.sim.getObject('/WALL3'),
            self.sim.getObject('/WALL4'),
            self.sim.getObject('/obstacle0'),
            self.sim.getObject('/obstacle1'),
            self.sim.getObject('/obstacle2')
        ]
        print("Connected and loaded all objects successfully!")

        #Define Action Space
        # - Discrete(N), e.g., 4 or 6 drone movement actions

        # we give the drone 4 basic moves: 0=forward, 1=backward, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        #Define Observation Space
        # - RGB camera image with shape (H, W, 3)
        # - dtype uint8
 
        # we shrink the camera image to 64x64 so the neural network trains faster
        self.IMG_H = 64
        self.IMG_W = 64
        
        # the observation is the raw RGB pixels from the drone's camera
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.IMG_H, self.IMG_W, 3), 
            dtype=np.uint8
        )

        #Simulation parameters
        # - Movement step size
        # - Drone fixed height
        # - Episode length limit

        # how much the drone moves in meters per action (e.g., 0.2 meters)
        self.delta = 0.2 
        
        # max steps before the drone gives up and restarts the episode
        self.max_steps = 150 
        
        # step counter to keep track of the current episode length
        self.current_step = 0
        # self.z_height = ...
        # self.steps = ... 

    # OBSERVATION FUNCTION
    # Must return an RGB image from the vision sensor.
    def _get_obs(self):
        """
        Retrieves the raw image from the CoppeliaSim vision sensor,
        converts it to a uint8 numpy array, resizes it to 64x64, 
        and returns the observation.
        """
        # grab the image from the vision sensor
        img, res = self.sim.getVisionSensorImg(self.camera)
        
        # convert the raw bytes into a numpy array (image matrix)
        img = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
        
        # CoppeliaSim images are flipped vertically by default, so we flip it back
        img = cv2.flip(img, 0)
        
        # resize to 64x64 as defined in our observation space
        img_resized = cv2.resize(img, (self.IMG_W, self.IMG_H))
        
        return img_resized


    # RESET FUNCTION
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # stop the CoppeliaSim simulation (blocking loop)
        self.sim.stopSimulation()
        
        # give it a small delay to make sure it stopped completely
        time.sleep(0.5) 
        
        # restart simulation cleanly
        self.sim.startSimulation()
        time.sleep(0.5)
        
        # reset drone position (fixed start)
        # we explicitly tell the drone to start at these coordinates
        drone_start_pos = [-3.0, -2.0, 0.5]
        self.sim.setObjectPosition(self.drone, self.sim.handle_world, drone_start_pos)
        
        # reset target position
        # It will spawn the target somewhere between Y: -1.0 and Y: 1.0
        #random_y = np.random.uniform(-1.0, 1.0)
        target_pos = [3.0, 0.0, 0.2] 
        self.sim.setObjectPosition(self.target, self.sim.handle_world, target_pos)
        
        # reset step counter
        self.current_step = 0
        
        # return the initial observation
        obs = self._get_obs()
        info = {}
        
        return obs, info

    # STEP FUNCTION
    def step(self, action):
        self.current_step += 1
        
        # apply movement based on 'action'
        pos = self.sim.getObjectPosition(self.drone, self.sim.handle_world)
        if action == 0: pos[0] += self.delta   # Forward
        elif action == 1: pos[0] -= self.delta # Backward
        elif action == 2: pos[1] += self.delta # Left
        elif action == 3: pos[1] -= self.delta # Right
        self.sim.setObjectPosition(self.drone, self.sim.handle_world, pos)
        
        # step the simulation forward
        # we give the simulation a moment to update physics and the camera frame
        time.sleep(0.05) 
        
        # collect observation via _get_obs()
        obs = self._get_obs()
        
        # compute reward based on your chosen logic
        # default penalty for each step to encourage faster paths
        reward = -0.5 
        
        target_pos = self.sim.getObjectPosition(self.target, self.sim.handle_world)
        dist = np.linalg.norm(np.array(pos[:2]) - np.array(target_pos[:2]))
        
        # check termination conditions (goal, collision, etc.)
        terminated = False
        truncated = False
        
        # condition A: Reached the goal
        if dist < 0.6:
            print(">>> Target Reached! <<<")
            reward = 100.0
            terminated = True
        else:
            # condition B: Hit a wall (collision)
            for wall in self.obstacles:
                res_collision, _ = self.sim.checkCollision(self.drone, wall)
                if res_collision == 1:
                    print(">>> Crash! Hit an obstacle. <<<")
                    reward = -50.0
                    terminated = True
                    break
                    
        # condition C: Ran out of time (max steps)
        if self.current_step >= self.max_steps:
            truncated = True
        
        # condition D: Out of Bounds (penalty)
        if abs(pos[0]) > 5.0 or abs(pos[1]) > 5.0:
            print(">>> Out of Bounds! Returning to center... <<<")
            reward = -50.0
            terminated = True
            
       
        # return: (obs, reward, terminated, truncated, info)
        info = {}
        return obs, float(reward), terminated, truncated, info



# =======================
# Optional: main block
# =======================
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    print("--- Initializing Drone Environment ---")
    env = DroneEnv()
    
    print("--- Checking Environment Compatibility ---")
    check_env(env, warn=True)

    print("--- Creating PPO Model ---")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        batch_size=64,
        n_steps=2048,
        learning_rate=3e-4,
    )

    print("--- Starting Training ---")
    #model.learn(total_timesteps=5000)
    model.learn(total_timesteps=100000)

    model.save("ppo_drone_navigation")
    print("Model saved! Training Complete!")

    pass
