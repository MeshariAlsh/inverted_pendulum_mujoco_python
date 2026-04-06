
# Code from: https://stable-baselines3.readthedocs.io/en/master/guide/plotting.html#

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, window_func

# Load the results
log_dir = "logs/PPO/"
df = load_results(log_dir)

# Convert dataframe (x=timesteps, y=episodic return)
x, y = ts2xy(df, "timesteps")

# Plot raw data
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.scatter(x, y, s=2, alpha=0.6)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Raw Episode Rewards")

# Plot smoothed data with custom window
plt.subplot(2, 1, 2)
if len(x) >= 50:  # Only smooth if we have enough data
    x_smooth, y_smooth = window_func(x, y, 50, np.mean)
    plt.plot(x_smooth, y_smooth, linewidth=2)
    plt.xlabel("Timesteps")
    plt.ylabel("Average Episode Reward (50-episode window)")
    plt.title("Smoothed Episode Rewards")

plt.tight_layout()
plt.show()

