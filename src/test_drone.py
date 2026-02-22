import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from drone_env import DroneEnv 
import time

def test_ai():
    print("--- Loading Environment and Trained Model ---")
    # initialize the environment
    env = DroneEnv()
    
    # load the saved model (ppo_drone_navigation.zip)
    model = PPO.load("models/ppo_drone_navigation", env=env)
    
    print("--- Starting Test Flights (5 episodes) ---")
    
    for episode in range(1, 6):
        obs, info = env.reset()
        terminated = False
        truncated = False
        score = 0
        
        print(f"Episode: {episode}")
        
        while not (terminated or truncated):
            # AI chooses the best action based on what it learned
            action, _states = model.predict(obs, deterministic=True)
            
            # apply on the action
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            
            # slow down a bit so we can watch it in CoppeliaSim
            time.sleep(0.05)
            
        print(f"Final Score for Episode {episode}: {score}")

    print("--- Test Finished ---")
    env.sim.stopSimulation()

if __name__ == "__main__":

    test_ai()
