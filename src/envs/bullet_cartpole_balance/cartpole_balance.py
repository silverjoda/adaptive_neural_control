import os
import time

import numpy as np
import pybullet as p
import pybullet_data
import torch as T
from gym import spaces


class CartPoleBulletEnv():
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, config):
        self.seed = config["seed"]
        if self.seed is not None:
            np.random.seed(self.seed)
            T.manual_seed(self.seed)
        else:
            rnd_seed = int((time.time() % 1) * 10000000)
            np.random.seed(rnd_seed)
            T.manual_seed(rnd_seed + 1)

        self.config = config

        self.obs_dim = 4
        self.act_dim = 1

        # Episode variables
        self.step_ctr = 0

        if (self.config["animate"]):
            self.client_ID = p.connect(p.GUI)
        else:
            self.client_ID = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setTimeStep(self.config["sim_timestep"])
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.robot = None
        self.robot = self.load_robot()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        self.cartpole = self.load_robot()

        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

    def load_robot(self):
        # Remove old robot
        if self.robot is not None:
            p.removeBody(self.robot)
        robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]))
        return robot

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

        return [x, x_dot, theta_1, theta_dot_1]

    def render(self, close=False):
        pass

    def step(self, ctrl):
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=ctrl * 100)
        p.stepSimulation()

        self.step_ctr += 1
        pendulum_height = p.getLinkState(self.cartpole, 1)[0][2]

        # x, x_dot, theta, theta_dot
        obs = self.get_obs()
        x, x_dot, theta_1, theta_dot_1 = obs

        height_rew = pendulum_height
        ctrl_pen = np.square(ctrl[0]) * 0.001
        r = height_rew - abs(x) * 0.1 - ctrl_pen

        done = self.step_ctr > self.config["max_steps"] or pendulum_height < 0.2

        return np.array(obs), r, done, {}

    def reset(self):
        self.step_ctr = 0
        p.resetJointState(self.cartpole, 0, targetValue=0, targetVelocity=np.random.randn() * 0.001)
        p.resetJointState(self.cartpole, 1, targetValue=0, targetVelocity=np.random.randn() * 0.001)
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
            for j in range(self.config["max_steps"]):
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
            for act in [[0], [-.3], [.3]]:
                for j in range(120):
                    obs, _, _, _ = self.step(np.array(act))
                    time.sleep(0.02)
                    print(obs)

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    import yaml
    with open("configs/cartpole_balance_config.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = CartPoleBulletEnv(env_config)
    env.demo()