import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)

import math
import numpy as np
import pybullet as p

import cv2
import time
import socket
import torch as T

import gym
from gym import spaces
from gym.utils import seeding

class DoubleCartPoleBulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, animate=False, action_input=False, max_steps=200, seed=None):
        if (animate):
          p.connect(p.GUI)
        else:
          p.connect(p.DIRECT)

        if seed is not None:
            np.random.seed(seed)
            T.manual_seed(seed)

        self.animate = animate
        self.prev_act_input = action_input

        # Simulator parameters
        self.max_steps = max_steps
        self.obs_dim = 6 + 1 + int(self.prev_act_input)
        self.act_dim = 1
        self.timeStep = 0.02

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        self.target_var = 2.0
        self.target_change_prob = 0.008
        self.mass_min = 1.0
        self.mass_range = 0

        self.cartpole = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "double_cartpole_goal.urdf"))
        self.target_vis = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "target.urdf"))

        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.obs_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))


    def get_obs(self):
        # Get cartpole observation
        x, x_dot, theta_1, theta_dot_1, theta_2, theta_dot_2 = \
            p.getJointState(self.cartpole, 0)[0:2] + \
            p.getJointState(self.cartpole, 1)[0:2] + \
            p.getJointState(self.cartpole, 2)[0:2]

        # Get goal target observation
        x_goal, y_goal = p.getBasePositionAndOrientation(self.target_vis)[0][0:2]

        # Clip velocities
        x_dot = np.clip(x_dot / 5, -5, 5)
        theta_dot_1 = np.clip(theta_dot_1 / 5, -5, 5)
        theta_dot_2 = np.clip(theta_dot_2 / 5, -5, 5)

        # Change theta range to [-pi, pi]
        if theta_1 > 0:
            if theta_1 % (2 * np.pi) <= np.pi:
                theta_1 = theta_1 % (2 * np.pi)
            else:
                theta_1 = -np.pi + theta_1 % np.pi
        else:
            if theta_1 % (-2 * np.pi) >= -np.pi:
                theta_1 = theta_1 % (-2 * np.pi)
            else:
                theta_1 = np.pi + theta_1 % -np.pi

        theta_1 /= np.pi

        if theta_2 > 0:
            if theta_2 % (2 * np.pi) <= np.pi:
                theta_2 = theta_2 % (2 * np.pi)
            else:
                theta_2 = -np.pi + theta_2 % np.pi
        else:
            if theta_2 % (-2 * np.pi) >= -np.pi:
                theta_2 = theta_2 % (-2 * np.pi)
            else:
                theta_2 = np.pi + theta_2 % -np.pi

        theta_2 /= np.pi

        self.state = np.array([x, x_dot, theta_1, theta_dot_1, theta_2, theta_dot_2, x_goal])

        return self.state


    def render(self, close=False):
        pass


    def step(self, ctrl):
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=ctrl * 100)
        p.stepSimulation()

        self.step_ctr += 1
        tip_x, tip_y, tip_z = p.getLinkState(self.cartpole, 2)[0]
        tip_x_dot = p.getLinkState(self.cartpole, 2, 1)[6][0]

        obs = self.get_obs()

        target_rew = 1.0 / (1.0 + 10 * np.abs(tip_x - self.target))
        height_rew = tip_z * 0.3
        vel_pen = np.square(tip_x_dot)
        ctrl_pen = np.square(ctrl[0]) * 0.000
        r = height_rew + target_rew - ctrl_pen

        #p.removeAllUserDebugItems()
        #p.addUserDebugText("Tip: % 3.3f, % 3.3f, % 3.3f, % 3.2f" % (tip_x, tip_y, tip_z, tip_x_dot), [-1, 0, 2])
        #p.addUserDebugText("target_rew: % 3.3f, height_rew % 3.3f, vel_pen % 3.3f, ctrl_pen % 3.2f" % (target_rew, height_rew, vel_pen, ctrl_pen), [-2, 0, 2.5])

        done = self.step_ctr > self.max_steps or tip_z < 0.5

        # Change target
        if np.random.rand() < self.target_change_prob:
            self.target = np.clip(np.random.rand() * 2 * self.target_var - self.target_var, -2, 2)
            p.resetBasePositionAndOrientation(self.target_vis, [self.target, 0, 1], [0, 0, 0, 1])

        if self.prev_act_input:
            obs = np.concatenate((obs, ctrl))

        return obs, r, done, {}


    def reset(self):
        self.step_ctr = 0
        self.target = np.random.rand() * 2 * self.target_var - self.target_var
        p.resetBasePositionAndOrientation(self.target_vis, [self.target, 0, 1], [0, 0, 0, 1])

        self.mass_1, self.mass_2 = self.mass_min + np.random.rand(2) * self.mass_range

        p.resetJointState(self.cartpole, 0, targetValue=0, targetVelocity=0)
        p.resetJointState(self.cartpole, 1, targetValue=0, targetVelocity=0)
        p.resetJointState(self.cartpole, 2, targetValue=0, targetVelocity=0)
        p.changeDynamics(self.cartpole, 1, mass=self.mass_1)
        p.changeDynamics(self.cartpole, 2, mass=self.mass_2)
        p.changeVisualShape(self.cartpole, 1, rgbaColor=[self.mass_1 / (self.mass_min + self.mass_range),
                                                         1 - self.mass_1 / (self.mass_min + self.mass_range), 0, 1])
        p.changeVisualShape(self.cartpole, 2, rgbaColor=[self.mass_2 / (self.mass_min + self.mass_range),
                                                         1 - self.mass_2 / (self.mass_min + self.mass_range), 0, 1])
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 2, p.VELOCITY_CONTROL, force=0)
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
                nn_obs = T.FloatTensor(obs.astype(np.float32)).unsqueeze(0)
                action = policy(nn_obs).detach()
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
                nn_obs = T.FloatTensor(obs.astype(np.float32)).unsqueeze(0)
                action, h = policy((nn_obs.unsqueeze(0), h))
                obs, r, done, od, = self.step(action[0][0].detach().numpy())
                cr += r
                total_rew += r
                if slow:
                    time.sleep(0.01)
            print("Total episode reward: {}".format(cr))
        print("Total reward: {}".format(total_rew))


    def demo(self):
        for i in range(100):
            self.reset()
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                obs, _, _, _ = self.step(np.array([-1]))
                input("Press Enter to continue...")
                print(obs)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                obs, _, _, _ = self.step(np.array([1]))
                time.sleep(0.02)
                print(obs)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([-0.3]))
                time.sleep(0.02)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([0.3]))
                time.sleep(0.02)


    def kill(self):
        p.disconnect()

    def close(self):
        self.kill()


if __name__ == "__main__":
    env = DoubleCartPoleBulletEnv(animate=True)
    env.demo()