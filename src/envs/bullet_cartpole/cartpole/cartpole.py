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

if socket.gethostname() != "goedel":
    import gym
    from gym import spaces
    from gym.utils import seeding

class CartPoleBulletEnv():
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, animate=False, max_steps=200, seed=None):
        if (animate):
          p.connect(p.GUI)
        else:
          p.connect(p.DIRECT)

        if seed is not None:
            np.random.seed(seed)
            T.manual_seed(seed)

        self.animate = animate

        # Simulator parameters
        self.max_steps = max_steps
        self.obs_dim = 4
        self.act_dim = 1
        self.timeStep = 0.02

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        self.cartpole = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cartpole.urdf"))

        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.obs_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))


    def get_obs(self):
        # Get cartpole observation
        x, x_dot, theta_1, theta_dot_1 = \
            p.getJointState(self.cartpole, 0)[0:2] + \
            p.getJointState(self.cartpole, 1)[0:2]

        # Clip velocities
        x_dot = np.clip(x_dot / 5, -5, 5)
        theta_dot_1 = np.clip(theta_dot_1 / 5, -5, 5)

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

        self.state = np.array([x, x_dot, theta_1, theta_dot_1])

        return self.state


    def render(self, close=False):
        pass


    def step(self, ctrl):
        ctrl = np.clip(ctrl, -1, 1)
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=ctrl * 50)
        p.stepSimulation()

        self.step_ctr += 1
        pendulum_height = p.getLinkState(self.cartpole, 1)[0][2]

        # x, x_dot, theta, theta_dot
        obs = self.get_obs()
        x, x_dot, theta_1, theta_dot_1 = obs

        height_rew = pendulum_height
        r = height_rew + abs(x) * 0.1

        done = self.step_ctr > self.max_steps or pendulum_height < 0

        return obs, r, done, {}


    def reset(self):
        self.step_ctr = 0
        p.resetJointState(self.cartpole, 0, targetValue=0, targetVelocity=0)
        p.resetJointState(self.cartpole, 1, targetValue=0, targetVelocity=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
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
    env = CartPoleBulletEnv(animate=True)
    env.demo()