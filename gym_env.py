import gymnasium as gym 
from typing import Optional
import numpy as np
import math
import mujoco 


XML = """
<mujoco>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1" width="256" height="256"/>
        <texture name="checkard" type="2d" gridsize= "4 1" builtin="checker" rgb1="0 0 0 " rgb2=".5 .5 .5"    width="256" height="256" />
        <material name="checkard" texture="checkard" texuniform="true"/>
    </asset>

    <worldbody>
        <light diffuse=" .5 .5 .5" pos=" 0 0 3" dir="0 0 -1" />
        <geom type="plane" material="checkard" size=" 3 3 3" />

        <body pos="0 0 .20">
            <joint name="slide" type="slide" axis=" 1 0 0"/>
            <geom type="box" size=" .2 .2 .2" rgba=" 0 .9 .9 1"/>
            <body>
                <joint name="pendulum" type="hinge" axis=" 0 1 0"/>
                <geom type="cylinder" rgba=" 0 .9 .5 1" fromto="0 0 0.1 0 0 .6" size="0.04"/>  
            </body> 
        </body>
    </worldbody>

    <actuator>
        <motor joint="slide" name="slide" gear="100"/>
    </actuator>
</mujoco>
"""

class Inverted_Pendulum_env(gym.Env):

    def __init__(self, MJCF=XML):
        #intialise model and data
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)

        # Actions
        self.action_space = gym.spaces.Box(-3.0, 3.0, (1,), dtype=np.float32)
        
        # Observations 
        self.observation_space = gym.spaces.Box(-math.inf, math.inf, (4,), dtype=np.float32)

        self.intial_pole_angle =  np.random.uniform(-0.01, 0.01)  # For reward shaping

        # Reward 
        self.reward = 0 # Intial zero 

    def get_obs(self):
        """ Return the observation state of the pendulum i.e. the angle, velocity, and angle < 0.2"""
        
        observation = np.concatenate([self.data.qpos, self.data.qvel])

        return observation.astype(np.float32)
    

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = np.random.uniform(-0.01, 0.01)

        
        observation = self.get_obs()
        info = {}
    
        return observation, info
    
    def step(self, action):

        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Action will come from PPO 
        self.data.ctrl[0] = action[0]

        # Run a single step in the Mujoco simulation
        mujoco.mj_step(self.model, self.data)

        observation = self.get_obs()


        current_pole_angle = observation[1] # data.qpos 
        distance_pole = abs(current_pole_angle - self.intial_pole_angle)
        terminated = bool(abs(distance_pole) > 0.2)

        reward = (1 - distance_pole) if not terminated else 0

        # no use for them but it needs to be returned becuase  it's required by gym api
        truncated = False
        info = {}

        return  observation, reward, terminated, truncated, info
        