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

class JoyController:
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
        turn, vel = [self.joystick.get_axis(3), self.joystick.get_axis(1)]
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

        self.config = config

        if (self.config["animate"]):
            self.client_ID = p.connect(p.GUI)
        else:
            self.client_ID = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setTimeStep(self.config["sim_timestep"])
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.wheels = [8] # 15
        self.tires = [1, 3, 12, 14]
        self.inactive_wheels = []
        self.steering = [0, 2] # [4, 6]

        self.robot = self.load_robot()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        for j in range(p.getNumJoints(self.robot)):
            print("joint[", j, "]=", p.getJointInfo(self.robot, j))
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            p.getJointInfo(self.robot, j)

        c = p.createConstraint(self.robot, 9, self.robot, 11, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = p.createConstraint(self.robot, 10, self.robot, 13, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = p.createConstraint(self.robot, 9, self.robot, 13, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        # c = p.createConstraint(self.robot, 16, self.robot, 18, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
        #                        parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        # p.changeConstraint(c, gearRatio=1, maxForce=10000)
        #
        # c = p.createConstraint(self.robot, 16, self.robot, 19, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
        #                        parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        # p.changeConstraint(c, gearRatio=-1, maxForce=10000)
        #
        # c = p.createConstraint(self.robot, 17, self.robot, 19, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
        #                        parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        # p.changeConstraint(c, gearRatio=-1, maxForce=10000)
        #
        # c = p.createConstraint(self.robot, 1, self.robot, 18, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
        #                        parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        # p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        # c = p.createConstraint(self.robot, 3, self.robot, 19, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
        #                        parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        # p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

        for wheel in self.wheels:
            p.setJointMotorControl2(self.robot, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        for wheel in self.inactive_wheels:
            p.setJointMotorControl2(self.robot, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        self.JOYStick = JoyController(self.config)


    def load_robot(self):
        # Remove old robot
        if not hasattr(self, 'robot'): # "f10_racecar/racecar_differential.urdf"
            self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "f10_racecar/racecar_differential.urdf"),
                               physicsClientId=self.client_ID)

        # Change params
        p.changeDynamics(self.robot, -1, mass=self.config["mass"])
        for tire in self.tires:
            p.changeDynamics(self.robot, tire,
                             lateralFriction=self.config["lateralFriction"],
                             physicsClientId=self.client_ID)
        return self.robot

    def step(self, ctrl_raw):
        for wheel in self.wheels:
            p.setJointMotorControl2(self.robot,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity= ctrl_raw[0] * self.config["velocity_scalar"],
                                    force=self.config["max_force"])

        for steer in self.steering:
            p.setJointMotorControl2(self.robot, steer, p.POSITION_CONTROL, targetPosition=ctrl_raw[1] * self.config["steering_scalar"])

        p.stepSimulation()

    def demo(self):
        acts = [[1,0], [-1,0], [1,-1], [-1, 1]]
        for i in range(100):
            act = acts[i % 4] # np.random.rand(2) * 2 - 1

            for i in range(self.config["max_steps"]):
                self.step(act)
                time.sleep(self.config["sim_timestep"])

    def demo_joystick(self):
        while True:
            if self.config["controller_source"] == "joystick":
                vel, turn, b_x = self.JOYStick.get_joystick_input()
            else:
                vel, turn, b_x = [0,0,0]

            self.step([vel, turn])
            time.sleep(self.config["sim_timestep"])

    def test_motors(self):
        # acts = [[1,-0.5], [-1,0], [1,0.5], [0.5,1]]
        acts = [[1, 0], [-1, 0], [-1, -1], [0.5, 1]]
        for act in acts:
            for i in range(150):
                self.step(act)

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
    env.demo_joystick()