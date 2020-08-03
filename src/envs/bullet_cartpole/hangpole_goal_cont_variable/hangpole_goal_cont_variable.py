import os
import time

import gym
import numpy as np
import pybullet as p
from gym import spaces

import src.my_utils as my_utils


class HangPoleGoalContVariableBulletEnv(gym.Env):
    def __init__(self, animate=False, max_steps=200, action_input=False, latent_input=False):
        if (animate):
          p.connect(p.GUI)
        else:
          p.connect(p.DIRECT)

        self.animate = animate
        self.latent_input = latent_input
        self.action_input = action_input

        # Simulator parameters
        self.max_steps = max_steps
        self.animate = animate
        self.latent_dim = 1
        self.obs_dim = 4 + self.latent_dim * int(self.latent_input) + int(self.action_input) + 1
        self.act_dim = 1

        self.timeStep = 0.02

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        self.target_debug_line = None
        self.target_var = 1.2
        self.target_change_prob = 0.008
        self.weight_position_min = 0.0
        self.weight_position_var = 1.0

        self.cartpole = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hangpole_goal_cont_variable.urdf"))
        self.target_vis = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "target.urdf"))

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

    def get_obs(self):
        x, x_dot, theta, theta_dot = p.getJointState(self.cartpole, 0)[0:2] + p.getJointState(self.cartpole, 1)[0:2]

        # Clip velocities
        x_dot = np.clip(x_dot / 10, -7, 7)
        theta_dot = np.clip(theta_dot / 10, -7, 7)

        # Change operational point so that the jump in angle sign is down below
        theta += np.pi

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


    def get_latent_label(self):
        weight_pos = (2 * self.weight_position - 2 * self.weight_position_min) / self.weight_position_var - 1
        return np.array((weight_pos))

    def render(self, close=False):
        pass

    def step(self, ctrl):
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=ctrl * 30)
        p.setJointMotorControl2(self.cartpole, 2, p.POSITION_CONTROL, self.weight_position)
        p.stepSimulation()

        self.step_ctr += 1

        obs = self.get_obs()
        x, x_dot, theta, theta_dot = obs
        x_sphere = p.getLinkState(self.cartpole, 1)[0][0]
        x_dot_sphere = p.getLinkState(self.cartpole, 1, 1)[6][0]

        target_rew = 1.0 / (1.0 + 5 * np.abs(x_sphere - self.target))  # Reward agent for being close to target
        vel_pen = np.square(x_dot_sphere)  # Velocity pen
        ctrl_pen = np.square(ctrl[0]) * 0.001
        r = target_rew / (1 + 2.0 * vel_pen) - ctrl_pen  # Agent is rewarded only if low velocity near target

        done = (self.step_ctr > self.max_steps) or abs(theta) > 0.5

        #p.removeAllUserDebugItems()
        #p.addUserDebugText("Theta: % 3.3f" % (theta), [-1, 0, 2])
        #time.sleep(0.01)

        # Change target
        if np.random.rand() < self.target_change_prob:
            self.target = np.clip(np.random.rand() * 2 * self.target_var - self.target_var, -2, 2)
            p.resetBasePositionAndOrientation(self.target_vis, [self.target, 0, -self.weight_position], [0, 0, 0, 1]) # Weight position is here only to draw correct height of target

        if self.latent_input:
            obs = np.concatenate((obs, self.get_latent_label()))
        if self.action_input:
            obs = np.concatenate((obs, ctrl))

        obs = np.concatenate((obs, [self.target]))

        return obs, r, done, {}


    def reset(self):
        self.step_ctr = 0
        self.theta_prev = 1

        self.weight_position = self.weight_position_min + np.random.rand() * self.weight_position_var
        self.cartpole_mass = 5.0

        self.target = np.random.rand() * 2 * self.target_var - self.target_var
        p.resetBasePositionAndOrientation(self.target_vis, [self.target, 0, -self.weight_position], [0, 0, 0, 1])

        self.target_dist_prev = self.target

        p.resetJointState(self.cartpole, 0, targetValue=0, targetVelocity=0)
        p.resetJointState(self.cartpole, 1, targetValue=0, targetVelocity=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        #p.setJointMotorControl2(self.cartpole, 2, p.POSITION_CONTROL, self.weight_position)
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
            p.resetJointState(self.cartpole, 1, targetValue=np.pi, targetVelocity=0)
            acts = [0.1, -0.1, 0.3, -0.3]
            for a in acts:
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                for i in range(self.max_steps):
                    self.step(np.array([a]))
                    time.sleep(0.01)

    def kill(self):
        p.disconnect()



if __name__ == "__main__":
    env = HangPoleGoalContVariableBulletEnv(animate=True)
    env.demo()