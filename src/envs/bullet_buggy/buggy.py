import os
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import torch as T

import src.my_utils as my_utils

class BuggyBulletEnv(gym.Env):
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

        self.obs_dim = 9
        self.act_dim = 2

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

        self.observation_space = spaces.Box(low=np.array([-1] * self.obs_dim), high=np.array([1] * self.obs_dim))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

        self.inactive_wheels = [3, 5, 7]
        self.wheels = [2]
        self.steering = [4, 6]

        for wheel in self.inactive_wheels:
            p.setJointMotorControl2(self.robot, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        # self.targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -100, 100, 0)
        # self.maxForceSlider = p.addUserDebugParameter("maxForce", 0, 100, 10)
        # self.steeringSlider = p.addUserDebugParameter("steering", -0.5, 0.5, 0)

        self.target_A = self.target_B = None
        self.target_visualshape = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                      radius=0.1,
                                                      rgbaColor=[1,0,0,1],
                                                      physicsClientId=self.client_ID)
        self.update_targets()


    def load_robot(self):
        # Remove old robot
        if self.robot is not None:
            p.removeBody(self.robot)

        # Randomize robot params
        self.robot_params = {"mass": 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "wheel_base" : 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "wheel_width": 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "front_wheels_friction": 0.5 + np.random.rand() * 2.5 * self.config["randomize_env"],
                             "motor_force_multiplier": 50 + np.random.rand() * 30 * self.config["randomize_env"]}

        if not self.config["randomize_env"]:
            robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]),
                               physicsClientId=self.client_ID)
            return robot

        # Write params to URDF file
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]), "r") as in_file:
            buf = in_file.readlines()

        index = self.config["urdf_name"].find('.urdf')
        output_urdf = self.config["urdf_name"][:index] + '_rnd' + self.config["urdf_name"][index:]

        # Load urdf
        robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]), physicsClientId=self.client_ID)

        # Change params
        #p.changeDynamics(self.robot, -1, mass=self.robot_params["mass"])
        return robot

    def get_obs(self):
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)
        return torso_pos, torso_quat, torso_vel, torso_angular_vel

    def update_targets(self):
        if self.target_A is None:
            self.target_A = np.random.rand(2) * self.config["target_dispersal_distance"] - self.config[
                "target_dispersal_distance"] / 2
            self.target_B = np.random.rand(2) * self.config["target_dispersal_distance"] - self.config[
                "target_dispersal_distance"] / 2

            self.target_A_body = p.createMultiBody(baseMass=0,
                                                   baseVisualShapeIndex=self.target_visualshape,
                                                   basePosition=[self.target_A[0], self.target_A[1], 0],
                                                   physicsClientId=self.client_ID)
            self.target_B_body = p.createMultiBody(baseMass=0,
                                                   baseVisualShapeIndex=self.target_visualshape,
                                                   basePosition=[self.target_B[0], self.target_B[1], 0],
                                                   physicsClientId=self.client_ID)
        else:
            self.target_A = self.target_B
            self.target_B = np.random.rand(2) * self.config["target_dispersal_distance"] - self.config[
                "target_dispersal_distance"] / 2
            p.resetBasePositionAndOrientation(self.target_A_body, [self.target_A[0], self.target_A[1], 0], [0, 0, 0, 1], physicsClientId=self.client_ID)
            p.resetBasePositionAndOrientation(self.target_B_body, [self.target_B[0], self.target_B[1], 0], [0, 0, 0, 1], physicsClientId=self.client_ID)

    def render(self, close=False):
        pass

    def step(self, ctrl):
        # maxForce = p.readUserDebugParameter(self.maxForceSlider)
        # targetVelocity = p.readUserDebugParameter(self.targetVelocitySlider)
        # steeringAngle = p.readUserDebugParameter(self.steeringSlider)

        for wheel in self.wheels:
            p.setJointMotorControl2(self.robot,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=ctrl[0] * self.config["velocity_scaler"],
                                    force=self.config["max_force"])

        for steer in self.steering:
            p.setJointMotorControl2(self.robot, steer, p.POSITION_CONTROL, targetPosition=ctrl[1])

        p.stepSimulation()
        self.step_ctr += 1

        torso_pos, torso_quat, torso_vel, torso_angular_vel = self.get_obs()
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        # Check if the agent has reached a target
        target_dist = np.sqrt(np.square(torso_pos[0] - self.target_A[0]) ** 2 + np.square(torso_pos[1] - self.target_A[1]) ** 2)
        r = -target_dist

        if target_dist < self.config["target_proximity_threshold"]:
            self.update_targets()
        done = (self.step_ctr > self.config["max_steps"])

        obs = np.concatenate((torso_pos[0:2], [yaw], torso_vel[0:2], self.target_A, self.target_B))

        return obs, r, done, {}

    def reset(self):
        self.step_ctr = 0
        p.resetJointState(self.robot, 0, targetValue=0, targetVelocity=0)
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_ID)
        obs, _, _, _ = self.step(np.zeros(self.act_dim))
        return obs

    def demo(self):
        for i in range(100):
            act = np.random.rand(2) * 2 - 1

            for i in range(self.config["max_steps"]):
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
    with open("configs/buggy_config.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = BuggyBulletEnv(env_config)
    env.demo()