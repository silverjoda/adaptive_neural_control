import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import numpy as np
import pybullet as p

import numpy as np
import random
import cv2
import src.my_utils as my_utils
import time
import socket

if socket.gethostname() != "goedel" or False:
    import gym
    from gym import spaces
    from gym.utils import seeding

class HangPoleGoalBulletEnv():
    def __init__(self, animate=False, max_steps=100, action_input=False):
        if (animate):
          p.connect(p.GUI)
        else:
          p.connect(p.DIRECT)

        self.animate = animate
        self.action_input = action_input

        # Simulator parameters
        self.max_steps = max_steps
        self.obs_dim = 4 + int(self.action_input) + 1
        self.act_dim = 1

        self.timeStep = 0.02

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        self.target_debug_line = None
        self.target_var = 2.0
        self.target_change_prob = 0.003
        self.dist_var = 2
        self.mass_var = 0
        self.mass_min = 1.0

        self.cartpole = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hangpole.urdf"))
        self.target_vis = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "target.urdf"))

        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.obs_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))
        self.metadata = None

        #print(self.dist_var, self.mass_var)


    def get_obs(self):
        x, x_dot, theta, theta_dot = p.getJointState(self.cartpole, 0)[0:2] + p.getJointState(self.cartpole, 1)[0:2]

        # Clip velocities
        x_dot = np.clip(x_dot / 3, -7, 7)
        theta_dot = np.clip(theta_dot / 3, -7, 7)

        # Change theta range to [-pi, pi]
        if theta > 0:
            if theta % (2 * np.pi) <= np.pi:
                theta = theta % (2 * np.pi)
            else:
                theta = -np.pi + theta % np.pi
        else:
            if theta % (-2 * np.pi) >= -np.pi:
                theta = theta % (-2 * np.pi)
            else:
                theta = np.pi + theta % -np.pi

        theta /= np.pi

        self.state = np.array([x, x_dot, theta, theta_dot])
        return self.state


    def render(self, close=False):
        pass


    def step(self, ctrl):
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=ctrl * 20)
        p.stepSimulation()

        self.step_ctr += 1

        # x, x_dot, theta, theta_dot
        obs = self.get_obs()
        x, x_dot, theta, theta_dot = obs
        x_sphere = p.getLinkState(self.cartpole, 1)[0][0]
        x_dot_sphere = p.getLinkState(self.cartpole, 1, 1)[6][0]

        target_rew = 1.0 / (1.0 + np.abs(x_sphere - self.target)) # Reward agent for being close to target
        vel_pen = np.square(x_dot_sphere) # Velocity pen
        r = target_rew / (1 + 3.0 * vel_pen) # Agent is rewarded only if low velocity near target

        #p.removeAllUserDebugItems()
        #p.addUserDebugLine((x, 0, 0), (x_sphere, 0, -np.cos(p.getJointState(self.cartpole, 1)[0])))
        #p.addUserDebugText("sphere mass: {0:.3f}".format(self.mass), [0, 0, 2])
        #p.addUserDebugText("sphere x: {0:.3f}".format(x_sphere), [0, 0, 2])
        #p.addUserDebugText("sphere x_dot: {}".format((1 - 0.5 * abs(x_dot_sphere))), [-4.0, 0, 2.2])
        #p.addUserDebugText("cart pen: {0:.3f}".format(cart_pen), [0, 0, 2])
        #p.addUserDebugText("x: {0:.3f}".format(x), [0, 0, 2])
        #p.addUserDebugText("x_target: {0:.3f}".format(self.target), [0, 0, 2.2])
        #p.addUserDebugText("cart_pen: {0:.3f}".format(cart_pen), [0, 0, 2.4])

        done = self.step_ctr > self.max_steps

        # Change target
        if np.random.rand() < self.target_change_prob:
            self.target = np.clip(np.random.rand() * 2 * self.target_var - self.target_var, -2, 2)
            p.resetBasePositionAndOrientation(self.target_vis, [self.target, 0, -1], [0, 0, 0, 1])


        if self.action_input:
            obs = np.concatenate((obs, ctrl))

        obs = np.concatenate((obs, [self.target]))

        return obs, r, done, {}


    def reset(self):
        self.step_ctr = 0
        self.theta_prev = 1
        self.target = np.random.rand() * 2 * self.target_var - self.target_var
        p.resetBasePositionAndOrientation(self.target_vis, [self.target, 0, -1], [0, 0, 0, 1])

        self.mass = random.sample([1,5,15], 1)[0] # self.mass_min + np.random.rand() * self.mass_var

        p.resetJointState(self.cartpole, 0, targetValue=0, targetVelocity=0)
        p.resetJointState(self.cartpole, 1, targetValue=0, targetVelocity=0)
        p.changeDynamics(self.cartpole, 1, mass=self.mass)
        p.changeVisualShape(self.cartpole, 1, rgbaColor=[self.mass / (self.mass_min + self.mass_var), 1 - self.mass / (self.mass_min + self.mass_var), 0, 1])
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
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
        for i in range(30):
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
        for i in range(30):
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
                time.sleep(0.01)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([1]))
                time.sleep(0.005)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([-0.3]))
                time.sleep(0.01)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([0.3]))
                time.sleep(0.01)


    def kill(self):
        p.disconnect()

    def close(self):
        self.kill()



if __name__ == "__main__":
    env = HangPoleGoalBulletEnv(animate=True)
    env.demo()