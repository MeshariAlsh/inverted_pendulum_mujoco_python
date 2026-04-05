import gymnasium as gym
import mujoco.viewer
from stable_baselines3 import PPO
from gym_env import Inverted_Pendulum_env
from gym_env import XML
import time
import  matplotlib.pyplot as plt
import numpy as np

EPISODES = 50
# 1 second = 1 / 0.002 = 500 steps
MAX_STEPS = 30000 
TIMESTEP = 0.002

# Load the current model we want
with open("model.txt", "r") as file:
    PPO_model = file.read()
    print(f"\nEvaluating model file: {PPO_model} \n")


cumlatative_rewards = []
survival_time = []

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)

env = Inverted_Pendulum_env(XML) 
model = PPO.load(f"{PPO_model}", env)

for episode in range(EPISODES):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        terminated = False

        while not terminated and steps < MAX_STEPS:
            step_time = time.time()
            action, state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        survived_time = steps * TIMESTEP
        status = "SURVIVED" if steps == MAX_STEPS else "FELL"
        print(f"Episode {episode+1} | {status} | Time: {survived_time:.1f}s | Steps: {steps} | Reward: {episode_reward:.2f}")
    
        cumlatative_rewards.append(episode_reward)
        survival_time.append(steps * TIMESTEP)
        #print(env.model.opt.timestep) To get an estimate of the mujoco step time in seconds. 0.002   

episodes = np.arange(1, EPISODES+1)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.bar(episodes, survival_time, color='blue')
ax1.axhline(y=60, color='red', linestyle='--', label='60 seconds target')
ax1.set_title("Survival Time per Episode")
ax1.set_xlabel('Episode')
ax1.set_ylabel('Survival Time')
ax1.legend()

ax2.plot(episodes, cumlatative_rewards, color='blue', marker='o', markersize=3)
ax2.axhline(y=0, color='red', linestyle='--', label='Zero reward')
ax2.set_title("Cumalative Reward per Episode")
ax2.set_xlabel('Episode')
ax2.set_ylabel('Total reward')
ax2.legend()

plt.suptitle('Model Evaluation - 50 Episodes', fontsize=14)
plt.tight_layout()
plt.show()
