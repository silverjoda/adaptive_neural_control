import os
import math
import numpy as np
import pybullet as p

import numpy as np
import cv2
import src.my_utils as my_utils
import time
import socket

import gym
from gym import spaces
from gym.utils import seeding

class DoubleCartpoleSwingupGoalVariable():
    def __init__(self, animate=False, max_steps=300, action_input=False, latent_input=False):
        if (animate):
          p.connect(p.GUI)
        else:
          p.connect(p.DIRECT)

        self.animate = animate
        self.latent_input = latent_input
        self.action_input = action_input

        # Simulator parameters
        self.max_steps = max_steps
        self.latent_dim = 2
        self.obs_dim = 6 + self.latent_dim * int(self.latent_input) + int(self.action_input) + 1
        self.act_dim = 1

        self.timeStep = 0.02

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        self.target_debug_line = None
        self.target_var = 2.0
        self.target_change_prob = 0.008
        self.mass_min = 1.0
        self.mass_range = 7.0

        self.cartpole = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "double_cartpole_swingup_goal_variable.urdf"))
        self.target_vis = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "target.urdf"))

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,))
        # TODO: -1,1 MIGHT BE A PROBLEM! Velocities have a much larger range!
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

    def get_obs(self):
        x, x_dot, theta_1, theta_dot_1, theta_2, theta_dot_2 = p.getJointState(self.cartpole, 0)[0:2] + p.getJointState(self.cartpole, 1)[0:2] + p.getJointState(self.cartpole, 2)[0:2]

        # Clip velocities
        x_dot = np.clip(x_dot / 7, -7, 7)
        theta_dot_1 = np.clip(theta_dot_1 / 7, -7, 7)
        theta_dot_2 = np.clip(theta_dot_2 / 7, -7, 7)

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

        self.state = np.array([x, x_dot, theta_1, theta_dot_1, theta_2, theta_dot_2])
        return self.state


    def get_latent_label(self):
        m_1 = (2 * self.mass_1 - 2 * self.mass_min) / self.mass_range - 1
        m_2 = (2 * self.mass_2 - 2 * self.mass_min) / self.mass_range - 1
        return np.array([m_1, m_2])


    def render(self, close=False):
        pass


    def step(self, ctrl):
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=ctrl * 100)
        p.stepSimulation()

        self.step_ctr += 1

        pendulum_height = p.getLinkState(self.cartpole, 2)[0][2]

        # x, x_dot, theta, theta_dot
        obs = self.get_obs()
        x, x_dot, theta_1, theta_dot_1, theta_2, theta_dot_2 = obs
        x_sphere = p.getLinkState(self.cartpole, 2)[0][0]
        x_dot_sphere = p.getLinkState(self.cartpole, 2, 1)[6][0]

        target_rew = 1.0 / (1.0 + 5 * np.abs(x_sphere - self.target))  # Reward agent for being close to target
        height_rew = pendulum_height
        vel_pen = np.square(x_dot_sphere)  # Velocity pen
        ctrl_pen = np.square(ctrl[0]) * 0.01
        r = height_rew + target_rew / (1 + 2.0 * vel_pen) - ctrl_pen  # Agent is rewarded only if low velocity near target

        #p.removeAllUserDebugItems()
        #p.addUserDebugText("Pendulum height: % 3.3f, -abs(x) % 3.3f, the %3.3f" % (pendulum_height, - abs(x) * 0.2, theta_1), [-1, 0, 2])

        done = self.step_ctr > self.max_steps

        # Change target
        if np.random.rand() < self.target_change_prob:
            self.target = np.clip(np.random.rand() * 2 * self.target_var - self.target_var, -2, 2)
            p.resetBasePositionAndOrientation(self.target_vis, [self.target, 0, 1], [0, 0, 0, 1])

        if self.latent_input:
            obs = np.concatenate((obs, self.get_latent_label()))
        if self.action_input:
            obs = np.concatenate((obs, ctrl))

        obs = np.concatenate((obs, [self.target]))

        return obs, r, done, {}


    def reset(self):
        self.step_ctr = 0
        self.theta_prev = 1
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


    def render_line(self):
        if not self.animate:
            return
        p.removeAllUserDebugItems()
        self.target_debug_line = p.addUserDebugLine([self.target, 0, 0],
                                                    [self.target, 0, 0.5],
                                                    lineWidth=6,
                                                    lineColorRGB=[1, 0, 0])

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
            self.reset()
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([-1]))
                time.sleep(0.02)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([1]))
                time.sleep(0.02)
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


if __name__ == "__main__":
    env = DoubleCartpoleSwingupGoalVariable(animate=True)
    env.demo()