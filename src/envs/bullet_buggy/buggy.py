import os
import time
import math as m
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

        self.obs_dim = 10
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

        self.wheels = [2, 3, 5, 7]
        self.steering = [4, 6]

        self.robot = None
        self.robot = self.load_robot()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        self.observation_space = spaces.Box(low=np.array([-4,-1,-1,-1,-np.pi,-1, -4, -4, -4, -4]),
                                            high=np.array([4, 1, 1, 1, np.pi, 1, 4, 4, 4, 4]))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

        for wheel in self.wheels:
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

    def set_randomize_env(self, rnd):
        self.config["randomize_env"] = rnd

    def load_robot(self):
        # Remove old robot
        if self.robot is None:
            robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]),
                               physicsClientId=self.client_ID)
        else:
            robot = self.robot

        # Randomize robot params
        self.robot_params = {"mass": 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "wheel_base" : 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "wheel_width": 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "wheels_friction": 0.3 + np.random.rand() * 1.5 * self.config["randomize_env"],
                             "max_force": 0.7 + np.random.rand() * 0.7 * self.config["randomize_env"],
                             "velocity_scaler": 50 + np.random.rand() * 80 * self.config["randomize_env"]}

        # Change params
        p.changeDynamics(robot, -1, mass=self.robot_params["mass"])
        for w in self.wheels:
            p.changeDynamics(robot, w,
                             lateralFriction=self.robot_params["wheels_friction"],
                             physicsClientId=self.client_ID)
        return robot

    def _quat_to_euler(self, w, x, y, z):
        pitch =  -m.asin(2.0 * (x*z - w*y))
        roll  =  m.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z)
        yaw   =  m.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z)
        return (pitch, roll, yaw)

    def get_obs(self):
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)
        torso_euler = self._quat_to_euler(*torso_quat)
        return torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel

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
        time.sleep(0.01)

    def step(self, ctrl):
        # maxForce = p.readUserDebugParameter(self.maxForceSlider)
        # targetVelocity = p.readUserDebugParameter(self.targetVelocitySlider)
        # steeringAngle = p.readUserDebugParameter(self.steeringSlider)

        for wheel in self.wheels:
            p.setJointMotorControl2(self.robot,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=np.clip(ctrl[0], -1, 1) * self.robot_params["velocity_scaler"],
                                    force=self.robot_params["max_force"])

        for steer in self.steering:
            p.setJointMotorControl2(self.robot, steer, p.POSITION_CONTROL, targetPosition=np.clip(ctrl[1], -1, 1))

        p.stepSimulation()
        self.step_ctr += 1

        torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel = self.get_obs()
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        # Check if the agent has reached a target
        target_dist = np.sqrt(np.square(torso_pos[0] - self.target_A[0]) ** 2 + np.square(torso_pos[1] - self.target_A[1]) ** 2)
        r = np.clip((self.prev_target_dist - target_dist) * 10, -3, 3)

        if target_dist < self.config["target_proximity_threshold"]:
            self.update_targets()
            self.prev_target_dist = np.sqrt(np.square(torso_pos[0] - self.target_A[0]) ** 2 + np.square(torso_pos[1] - self.target_A[1]) ** 2)
        else:
            self.prev_target_dist = target_dist

        done = (self.step_ctr > self.config["max_steps"])
        obs = np.concatenate((torso_pos[0:2], torso_vel[0:2], torso_euler[2:], torso_angular_vel[2:], self.target_A, self.target_B)).astype(np.float32)

        return obs, r, done, {}

    def reset(self, force_randomize=None):
        self.prev_target_dist = 0
        if self.config["randomize_env"]:
            self.robot = self.load_robot()
        self.step_ctr = 0
        p.resetJointState(self.robot, 0, targetValue=0, targetVelocity=0)
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_ID)
        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel = self.get_obs()
        self.update_targets()
        self.prev_target_dist = np.sqrt(np.square(torso_pos[0] - self.target_A[0]) ** 2 + np.square(torso_pos[1] - self.target_A[1]) ** 2)

        return obs

    def demo(self):
        for i in range(100):
            act = np.random.rand(2) * 2 - 1
            self.reset()

            for i in range(self.config["max_steps"]):
                obs, r, done, _ = self.step(act)
                #print(obs)
                time.sleep(0.01)

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = BuggyBulletEnv(env_config)
    env.demo()