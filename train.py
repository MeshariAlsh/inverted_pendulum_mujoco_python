from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import gymnasium as gym 
import mujoco as mj
from gym_env import Inverted_Pendulum_env


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
model_PPO = PPO("MlpPolicy", env, verbose=1)

print(" Training ...")
model_PPO.learn(total_timesteps=500000)

model_PPO.save(f"{PPO_model}")
print("Training complete")
