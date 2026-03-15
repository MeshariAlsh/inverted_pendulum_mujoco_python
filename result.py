import gymnasium as gym
import mujoco.viewer
from stable_baselines3 import PPO
from gym_env import Inverted_Pendulum_env
from gym_env import XML
import time

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)

env = Inverted_Pendulum_env(XML) 
model = PPO.load("ppo_models/pendulum_ppo_honours_500k_steps.zip", env)

obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    start = time.time()

    while viewer.is_running() and time.time() - start < 60:
        step_time = time.time()

        action, state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        with viewer.lock():
         viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(env.data.time % 2)

        # update model state for GUI
        viewer.sync()

        time_until_next_step = env.model.opt.timestep - (time.time() - step_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        if terminated:
           obs, info = env.reset()
