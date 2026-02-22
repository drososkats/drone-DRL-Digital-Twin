# Autonomous UAV Navigation - Digital Twin Project

## Project Overview
This project implements a **Digital Twin** for an autonomous drone (UAV) using **Deep Reinforcement Learning**. The drone is trained to navigate through an "Urban Canyon" environment in **CoppeliaSim** to locate a fire source (red target), using visual input from an onboard camera.

## System Architecture
- **RSPU (Robot Software Processing Unit):** Python-based controller interfacing with the simulation via ZMQ Remote API.
- **Algorithm:** Proximal Policy Optimization (PPO) from Stable Baselines3.
- **Vision:** CNN (Convolutional Neural Network) policy for processing raw pixels from the UAV's Vision Sensor.
- **Action Space:** 4 discrete actions (Forward, Backward, Left, Right) at a constant altitude.

## Directory Structure
- `src/`: Contains the environment definition (`drone_env.py`) and the testing script (`test_drone.py`).
- `models/`: The trained PPO model (`ppo_drone_navigation.zip`).
- `scene/`: The CoppeliaSim scene file (`.ttt`).
- `media/`: Video demonstration and performance screenshots.

## Performance
- **Training Steps:** 100,352
- **Explained Variance:** 0.751
- **Success Criteria:** The agent successfully identifies and reaches the target by navigating through/around urban obstacles.

## How to Run
1. Launch **CoppeliaSim** and open `scene/drone_scene.ttt`.
2. Ensure the **ZMQ Remote API** is active.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the test script:
   ```bash
   cd src

   python test_drone.py
