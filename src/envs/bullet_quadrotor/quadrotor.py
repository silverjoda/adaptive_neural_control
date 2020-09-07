import os
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import torch as T

import src.my_utils as my_utils

# Variable params: Mass, boom length, motor "inertia", motor max_force
class QuadrotorBulletEnv(gym.Env):
    def __init__(self, config):
        self.seed = config["seed"]
        if self.seed is not None:
            np.random.seed(self.seed)
            T.manual_seed(self.seed)
        else:
            rnd_seed = int((time.time() % 1) * 10000000)
            np.random.seed(rnd_seed)
            T.manual_seed(rnd_seed + 1)

        self.animate = config["animate"]
        self.max_steps = config["max_steps"]
        self.latent_input = config["latent_input"]
        self.action_input = config["action_input"]
        self.is_variable = config["is_variable"]
        self.sim_timestep = config["sim_timestep"]
        self.propeller_parasitic_torque_coeff = config["propeller_parasitic_torque_coeff"]

        self.obs_dim = 10 + int(self.action_input) * 4  # Orientation quaternion, linear + angular velocities
        self.act_dim = 4 # Motor commands (4)
        self.parasitic_torque_dir_vec = [1,-1,-1,1]

        # Episode variables
        self.step_ctr = 0
        self.current_motor_velocity_vec = np.array([0,0,0,0])

        if (self.animate):
          self.client_ID = p.connect(p.GUI)
        else:
          self.client_ID = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setTimeStep(self.sim_timestep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.load_robot()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        self.observation_space = spaces.Box(low=np.array([-1] * self.obs_dim), high=np.array([1] * self.obs_dim))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

    def load_robot(self):
        # Randomize robot params
        self.robot_params = {"mass": 1, "boom": 0.2, "motor_inertia": 10.0, "motor_max_force": 1.0}

        # Write params to URDF file
        # ...

        # Load urdf
        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "quadrotor.urdf"), physicsClientId=self.client_ID)

    def get_obs(self):
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)
        return torso_pos, torso_quat, torso_vel, torso_angular_vel

    def update_motor_vel(self, ctrl):
        self.current_motor_velocity_vec = np.clip(self.current_motor_velocity_vec +
                                                  (np.array(ctrl) * 2 - np.ones(4)) / self.robot_params["motor_inertia"], 0, 1)

    def render(self, close=False):
        pass

    def step(self, ctrl):
        self.update_motor_vel(ctrl)
        for i in range(4):
            p.applyExternalForce(self.robot, linkIndex=i*2 + 1, forceObj=[0, 0, self.current_motor_velocity_vec[i]],
                                 posObj=[0, 0, 0], flags=p.LINK_FRAME)
            p.applyExternalTorque(self.robot, linkIndex=i * 2 + 1, torqueObj=[0, 0, self.current_motor_velocity_vec[i] * self.parasitic_torque_dir_vec[i]],
                                  flags=p.LINK_FRAME)
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
    import yaml
    with open("../../algos/SB/configs/quadrotor_config.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = QuadrotorBulletEnv(env_config)
    env.demo()