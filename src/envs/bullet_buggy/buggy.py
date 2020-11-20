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
import logging
import pygame
import random

class JoyController():
    def __init__(self, config):
        self.config = config
        logging.info("Initializing joystick controller")
        pygame.init()
        if self.config["controller_source"] == "joystick":
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info("Initialized gamepad: {}".format(self.joystick.get_name()))
        else:
            logging.info("No joystick found")
        logging.info("Finished initializing the joystick controller.")
        self.button_x_state = 0

    def get_joystick_input(self):
        pygame.event.pump()
        turn, vel = [self.joystick.get_axis(2), self.joystick.get_axis(1)]
        button_x = self.joystick.get_button(1)
        pygame.event.clear()

        # [-1, 1]
        turn = -turn
        vel = -vel

        # button_x only when upon press
        if self.button_x_state == 0 and button_x == 1:
            self.button_x_state = 1
            button_x = 1
        elif self.button_x_state == 1 and button_x == 0:
            self.button_x_state = 0
            button_x = 0
        elif self.button_x_state == 1 and button_x == 1:
            self.button_x_state = 1
            button_x = 0
        else:
            self.button_x_state = 0
            button_x = 0

        return vel, turn, button_x

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

        self.wheels = [2,3,5,7] # [2, 3, 5, 7]
        self.inactive_wheels = []
        self.steering = [4, 6]

        self.robot = self.load_robot()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)
        # yaw, x_dot, y_dot, yaw_dot, relative_target_A, relative_target_B
        self.observation_space = spaces.Box(low=np.array([-np.pi,-2.,-2.,-2., -4., -4., -4., -4.], dtype=np.float32),
                                            high=np.array([np.pi, 2., 2., 2., 4., 4., 4., 4.], dtype=np.float32))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

        for wheel in self.wheels:
            p.setJointMotorControl2(self.robot, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        for wheel in self.inactive_wheels:
            p.setJointMotorControl2(self.robot, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        self.JOYStick = JoyController(self.config)
        self.create_targets()

    def create_targets(self):
        self.target_A = self.target_B = None
        self.target_visualshape_A = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                        radius=0.1,
                                                        rgbaColor=[1, 0, 0, 1],
                                                        physicsClientId=self.client_ID)
        self.target_visualshape_B = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                        radius=0.1,
                                                        rgbaColor=[1, 1, 0, 1],
                                                        physicsClientId=self.client_ID)
        self.update_targets()

    def set_randomize_env(self, rnd):
        self.config["randomize_env"] = rnd

    def load_robot(self):
        # Remove old robot
        if hasattr(self, 'robot'):
            robot = self.robot
        else:
            robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]),
                               physicsClientId=self.client_ID)

        # Randomize robot params
        self.robot_params = {"mass": 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "wheel_base" : 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "wheel_width": 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "wheels_friction": 1.4 + np.random.rand() * 1.5 * self.config["randomize_env"],
                             "max_force": 2.0 + np.random.rand() * 0.7 * self.config["randomize_env"], # With 0.7 works great
                             "velocity_scaler": 100 + np.random.rand() * 80 * self.config["randomize_env"]} # With 50 works great

        # Change params
        p.changeDynamics(robot, -1, mass=self.robot_params["mass"])
        for w in self.wheels:
            p.changeDynamics(robot, w,
                             lateralFriction=self.robot_params["wheels_friction"],
                             physicsClientId=self.client_ID)
        return robot

    def get_obs(self):
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)
        torso_euler = p.getEulerFromQuaternion(torso_quat)
        return torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel

    def update_targets(self):
        if self.target_A is None:
            self.target_A = np.random.rand(2) * self.config["target_dispersal_distance"] - self.config[
                "target_dispersal_distance"] / 2
            self.target_B = np.random.rand(2) * self.config["target_dispersal_distance"] - self.config[
                "target_dispersal_distance"] / 2

            self.target_A_body = p.createMultiBody(baseMass=0,
                                                   baseVisualShapeIndex=self.target_visualshape_A,
                                                   basePosition=[self.target_A[0], self.target_A[1], 0],
                                                   physicsClientId=self.client_ID)
            self.target_B_body = p.createMultiBody(baseMass=0,
                                                   baseVisualShapeIndex=self.target_visualshape_B,
                                                   basePosition=[self.target_B[0], self.target_B[1], 0],
                                                   physicsClientId=self.client_ID)
        else:
            self.target_A = self.target_B
            self.target_B = np.random.rand(2) * self.config["target_dispersal_distance"] - self.config[
                "target_dispersal_distance"] / 2
            p.resetBasePositionAndOrientation(self.target_A_body, [self.target_A[0], self.target_A[1], 0], [0, 0, 0, 1], physicsClientId=self.client_ID)
            p.resetBasePositionAndOrientation(self.target_B_body, [self.target_B[0], self.target_B[1], 0], [0, 0, 0, 1], physicsClientId=self.client_ID)

    def render(self, close=False):
        time.sleep(self.config["sim_timestep"])

    def step(self, ctrl):
        wheel_action = np.clip(ctrl[0], -1, 1)
        for wheel in self.wheels:
            p.setJointMotorControl2(self.robot,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity= wheel_action * self.robot_params["velocity_scaler"],
                                    force=self.robot_params["max_force"])

        for steer in self.steering:
            p.setJointMotorControl2(self.robot, steer, p.POSITION_CONTROL, targetPosition=np.clip(ctrl[1], -1, 1))

        p.stepSimulation()
        self.step_ctr += 1

        torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel = self.get_obs()

        # Orientation reward
        tar_angle = np.arctan2(self.target_A[1] - torso_pos[1], self.target_A[0] - torso_pos[0])
        yaw_deviation = np.min((abs((torso_euler[2] % 6.283) - (tar_angle % 6.283)), abs(torso_euler[2] - tar_angle)))

        # Check if the agent has reached a target
        target_dist = np.sqrt((torso_pos[0] - self.target_A[0]) ** 2 + (torso_pos[1] - self.target_A[1]) ** 2)
        vel_rew = np.clip((self.prev_target_dist - target_dist) * 10, -3, 3)
        #heading_rew = np.clip((self.prev_yaw_deviation - yaw_deviation) * 3, -2, 2)
        r = vel_rew

        if target_dist < self.config["target_proximity_threshold"]:
            self.update_targets()
            self.prev_target_dist = np.sqrt((torso_pos[0] - self.target_A[0]) ** 2 + (torso_pos[1] - self.target_A[1]) ** 2)
            tar_angle = np.arctan2(self.target_A[1] - torso_pos[1], self.target_A[0] - - torso_pos[0])
            yaw_deviation = np.min((abs((torso_euler[2] % np.pi * 2) - (tar_angle % np.pi * 2)), abs(torso_euler[2] - tar_angle)))
            self.prev_yaw_deviation = yaw_deviation
        else:
            self.prev_target_dist = target_dist
            self.prev_yaw_deviation = yaw_deviation

        # Calculate relative positions of targets
        relative_target_A = self.target_A[0] - torso_pos[0], self.target_A[1] - torso_pos[1]
        relative_target_B = self.target_B[0] - torso_pos[0], self.target_B[1] - torso_pos[1]

        done = (self.step_ctr > self.config["max_steps"])
        obs = np.concatenate((torso_euler[2:3], torso_vel[0:2], torso_angular_vel[2:3], relative_target_A, relative_target_B)).astype(np.float32)

        return obs, r, done, {}

    def reset(self, force_randomize=None):
        if self.config["randomize_env"]:
            self.robot = self.load_robot()
        self.step_ctr = 0

        p.resetJointState(self.robot, 0, targetValue=0, targetVelocity=0)
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_ID)

        torso_pos, _, torso_euler, _, _ = self.get_obs()
        self.update_targets()
        self.prev_target_dist = np.sqrt((torso_pos[0] - self.target_A[0]) ** 2 + (torso_pos[1] - self.target_A[1]) ** 2)
        tar_angle = np.arctan2(self.target_A[1] - torso_pos[1], self.target_A[0] - torso_pos[0])
        yaw_deviation = np.min((abs((torso_euler[2] % np.pi * 2) - (tar_angle % np.pi * 2)), abs(torso_euler[2] - tar_angle)))
        self.prev_yaw_deviation = yaw_deviation

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs

    def demo(self):
        acts = [[1,0], [-1,0], [1,-1], [-1, 1]]
        for i in range(100):
            act = acts[i % 4] # np.random.rand(2) * 2 - 1
            self.reset()

            for i in range(self.config["max_steps"]):
                obs, r, done, _ = self.step(act)
                #print(obs)
                time.sleep(self.config["sim_timestep"])

    def test_motors(self):
        # acts = [[1,-0.5], [-1,0], [1,0.5], [0.5,1]]
        acts = [[1, 0], [-1, 0], [-1, -1], [0.5, 1]]
        self.reset()
        for act in acts:
            for i in range(150):
                obs, r, done, _ = self.step(act)
                self.render()


    def close(self):
        p.disconnect()


    def gather_data(self, policy=None, n_iterations=20000):
        # Initialize data lists
        data_position = []
        data_vel = []
        data_rotation = []
        data_angular_vel = []
        data_timestamp = []
        data_action = []

        obs = self.reset()

        print("Starting the control loop")
        try:
            for i in range(n_iterations):
                iteration_starttime = time.time()

                # Update sensor data
                position_rob, rotation_rob, euler_rob, vel_rob, angular_vel_rob = self.get_obs()

                if self.config["controller_source"] == "joystick":
                    # Read target control inputs
                    m_1, m_2, _ = self.JOYStick.get_joystick_input()
                else:
                    m_1, m_2 = policy(obs)

                data_position.append(position_rob)
                data_vel.append(vel_rob)
                data_rotation.append(rotation_rob)
                data_angular_vel.append(angular_vel_rob)
                data_timestamp.append(0)
                data_action.append([m_1, m_2])

                # Write control to servos
                obs = self.step([m_1, m_2])

                # Sleep to maintain correct FPS
                while time.time() - iteration_starttime < self.config["sim_timestep"]: pass
        except:
            print("Interrupted by user")

        # Save data
        prefix = os.path.join("data", time.strftime("%Y_%m_%d_"))
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        prefix = os.path.join(prefix, ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3)))

        data_position = np.array(data_position, dtype=np.float32)
        data_vel = np.array(data_vel, dtype=np.float32)
        data_rotation = np.array(data_rotation, dtype=np.float32)
        data_angular_vel = np.array(data_angular_vel, dtype=np.float32)
        data_timestamp = np.array(data_timestamp, dtype=np.float32)
        data_action = np.array(data_action, dtype=np.float32)

        np.save(prefix + "_position", data_position)
        np.save(prefix + "_vel", data_vel)
        np.save(prefix + "_rotation", data_rotation)
        np.save(prefix + "_angular_vel", data_angular_vel)
        np.save(prefix + "_timestamp", data_timestamp)
        np.save(prefix + "_action", data_action)

        print("Saved data")


if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = BuggyBulletEnv(env_config)
    #env.gather_data()
    env.demo()