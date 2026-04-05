import gymnasium as gym
import mujoco.viewer
from stable_baselines3 import PPO
from gym_env import Inverted_Pendulum_env
from gym_env import XML
import time
import numpy as np


# Load the current model we want 
with open("model.txt", "r") as file:
    PPO_model = file.read()
    print(f"\n Running model file: {PPO_model} \n")


m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)

env = Inverted_Pendulum_env(XML) 
model = PPO.load(f"{PPO_model}", env)

obs, info = env.reset()

last_randomised = 0

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    start = time.time()

    while viewer.is_running() and time.time() - start < 1000:
        step_time = time.time()
        elapsed = time.time() - start

        if elapsed - last_randomised  >= 5.0:
           env.data.qpos[1]  = np.random.uniform(-0.2, 0.2)
           last_randomised = elapsed


        action, state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        #print(f"Before Sync | Current Reward: {reward}")
        #print(f"Cart Position: {obs[0]}")
        #print(f"Cart Velocity: {obs[2]} \n")

        #print(f"Pole Angle: {obs[1]}")
        #print(f"Pole Velocity: {obs[3]} \n")

        with viewer.lock():
         viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(env.data.time % 2)

        # update model state for GUI
        viewer.sync()
        

        time_until_next_step = env.model.opt.timestep - (time.time() - step_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        if terminated:
           obs, info = env.reset()
           last_randomised = 0
           print(f"TERMINATION OCCURRED\n")
