from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import gymnasium as gym 
import mujoco as mj
from gym_env import Inverted_Pendulum_env
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter


# Create log directory
log_dir = "logs/PPO/"
os.makedirs(log_dir, exist_ok=True)

# Load the current model we want
with open("model.txt", "r") as file:
    PPO_model = file.read()
    print(f"\Training model file: {PPO_model} \n")

gym.register(
    id="Inverted_Pendulum_version_1",
    entry_point=Inverted_Pendulum_env,
    max_episode_steps=500,  
)

env = gym.make("Inverted_Pendulum_version_1")
env = Monitor(env, log_dir)

model_PPO = PPO("MlpPolicy", env, verbose=1)

print(" Training ...")
model_PPO.learn(total_timesteps=500000)

model_PPO.save(f"{PPO_model}")
print("Training complete")

# Plot the results
plot_results([log_dir], 20_000, results_plotter.X_TIMESTEPS, "PPO CartPole")
plt.show()
