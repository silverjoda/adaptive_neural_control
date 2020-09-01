import os
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

import src.my_utils as my_utils

# Variable params: Mass, boom length, motor "inertia", motor max_force
class QuadrotorBulletEnv(gym.Env):
    def __init__(self, animate=False, max_steps=300, action_input=False, latent_input=False, is_variable=False):
        if (animate):
          self.client_ID = p.connect(p.GUI)
        else:
          self.client_ID = p.connect(p.DIRECT)

        self.animate = animate
        self.latent_input = latent_input
        self.action_input = action_input
        self.is_variable = is_variable

        # Simulator parameters
        self.max_steps = max_steps
        self.animate = animate
        self.latent_dim = 3
        self.obs_dim = 10 + self.latent_dim * int(self.latent_input) + int(self.action_input) # Orientation quaternion, linear + angular velocities
        self.act_dim = 4 # Motor commands (4)

        self.timeStep = 0.02
        self.step_ctr = 0

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.load_robot()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        self.observation_space = spaces.Box(low=np.array([-1] * self.obs_dim), high=np.array([1] * self.obs_dim))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

    def load_robot(self):
        # Randomize robot params
        self.robot_params = {"mass": 1, "boom": 0.2, "motor_inertia": 1.0, "motor_max_force": 1.0}

        # Write params to URDF file
        # ...

        # Load urdf
        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "quadrotor.urdf"), physicsClientId=self.client_ID)

    def get_obs(self):
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)
        return torso_pos, torso_quat, torso_vel, torso_angular_vel

    def render(self, close=False):
        pass

    def step(self, ctrl):
        for i in range(4):
            p.applyExternalForce(self.robot, linkIndex=i*2 + 1, forceObj=[0, 0, ctrl[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME)
        p.stepSimulation()

        self.step_ctr += 1

        torso_pos, torso_quat, torso_vel, torso_angular_vel = obs = self.get_obs()

        r = 0

        done = (self.step_ctr > self.max_steps) # or abs(theta) > 0.6

        return obs, r, done, {}

    def reset(self):
        self.step_ctr = 0
        p.resetJointState(self.robot, 0, targetValue=0, targetVelocity=0)
        obs, _, _, _ = self.step(np.zeros(self.act_dim))
        return obs

    def test(self, policy, slow=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.render_prob = 1.0
        total_rew = 0
        for i in range(100):
            obs = self.reset()
            cr = 0
            for j in range(self.max_steps):
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0].numpy())
                cr += r
                total_rew += r
                if slow:
                    time.sleep(0.01)
            print("Total episode reward: {}".format(cr))
        print("Total reward: {}".format(total_rew))

    def test_recurrent(self, policy, slow=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        total_rew = 0
        for i in range(100):
            obs = self.reset()
            h = None
            cr = 0
            for j in range(self.max_steps):
                action, h = policy((my_utils.to_tensor(obs, True).unsqueeze(0), h))
                obs, r, done, od, = self.step(action[0][0].detach().numpy())
                cr += r
                total_rew += r
                if slow:
                    time.sleep(0.01)
            print("Total episode reward: {}".format(cr))
        print("Total reward: {}".format(total_rew))

    def demo(self):
        for i in range(100):
            act = np.array([0.1, 0.1, 0.1, 0.1]) * 44

            for i in range(self.max_steps):
                obs, r, done, _ = self.step(act)
                #print(obs)
                time.sleep(0.01)

            self.reset()

    def kill(self):
        p.disconnect()

    def close(self):
        self.kill()


if __name__ == "__main__":
    env = QuadrotorBulletEnv(animate=True)
    env.demo()