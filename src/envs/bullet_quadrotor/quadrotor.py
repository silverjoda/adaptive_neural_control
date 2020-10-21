import os
import time
import math as m
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import torch as T
import logging
import pygame

import src.my_utils as my_utils

class JoyController():
    def __init__(self, config):
        self.config = config
        logging.info("Initializing joystick controller")
        pygame.init()
        if self.config["target_vel_source"] == "joystick":
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info("Initialized gamepad: {}".format(self.joystick.get_name()))
        logging.info("No joystick found")
        logging.info("Finished initializing the joystick controller.")
        self.button_x_state = 0

    def get_joystick_input(self):
        pygame.event.pump()
        throttle, t_yaw, t_roll, t_pitch = \
            [self.joystick.get_axis(i) for i in range(4)]
        button_x = self.joystick.get_button(1)
        pygame.event.clear()

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

        return throttle, t_roll, t_pitch, t_yaw, button_x


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

        self.config = config
        self.obs_dim = 13
        self.act_dim = 4
        self.parasitic_torque_dir_vec = [1,-1,-1,1]

        # Episode variables
        self.step_ctr = 0
        self.current_motor_velocity_vec = np.array([0.,0.,0.,0.])

        if (config["animate"]):
          self.client_ID = p.connect(p.GUI)
        else:
          self.client_ID = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setTimeStep(config["sim_timestep"])
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.robot = None
        self.robot = self.load_robot()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        self.observation_space = spaces.Box(low=np.array([-1] * self.obs_dim), high=np.array([1] * self.obs_dim))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

        self.rnd_target_vel_source = my_utils.SimplexNoise(4)
        self.joystick_controller = JoyController(self.config)

    def set_randomize_env(self, rnd):
        self.config["randomize_env"] = rnd

    def load_robot(self):
        # Remove old robot
        if self.robot is not None:
            p.removeBody(self.robot)

        # Randomize robot params
        self.robot_params = {"mass": 1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "boom": 0.1 + np.random.rand() * 0.5 * self.config["randomize_env"],
                             "motor_inertia_coeff": 0.0 + np.random.rand() * 0.25 * self.config["randomize_env"],
                             "motor_force_multiplier": 30 + np.random.rand() * 20 * self.config["randomize_env"]}

        if not self.config["randomize_env"]:
            robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]),
                               physicsClientId=self.client_ID)
            return robot

        # Write params to URDF file
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]), "r") as in_file:
            buf = in_file.readlines()

        index = self.config["urdf_name"].find('.urdf')
        output_urdf = self.config["urdf_name"][:index] + '_rnd' + self.config["urdf_name"][index:]

        # Change link lengths in urdf
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_urdf), "w") as out_file:
            for line in buf:
                if "<cylinder radius" in line:
                    out_file.write(f'          <cylinder radius="0.015" length="{self.robot_params["boom"]}"/>\n')
                elif line.rstrip('\n').endswith('<!--boomorigin-->'):
                    out_file.write(f'        <origin xyz="0 {self.robot_params["boom"] / 2.} 0.0" rpy="-1.5708 0 0" /><!--boomorigin-->\n')
                elif line.rstrip('\n').endswith('<!--motorpos-->'):
                    out_file.write(f'      <origin xyz="0 {self.robot_params["boom"]} 0" rpy="0 0 0"/><!--motorpos-->\n')
                else:
                    out_file.write(line)

        # Load urdf
        robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_urdf), physicsClientId=self.client_ID)

        # Change base mass
        p.changeDynamics(robot, -1, mass=self.robot_params["mass"])

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

    def update_motor_vel(self, ctrl):
        self.current_motor_velocity_vec = np.clip(self.current_motor_velocity_vec * self.robot_params["motor_inertia_coeff"] +
                                                  np.array(ctrl) * (1 - self.robot_params["motor_inertia_coeff"]), 0, 1)

    def render(self, close=False):
        time.sleep(self.config["sim_timestep"])

    def calc_turbulence_coeffs(self, height_vec):
        height_vec_arr = np.array(height_vec)
        scaled_height_arr = np.clip(((height_vec_arr - 0.15) ** 2) / 0.15, 0, 0.15) * 2
        turb_coeff = (np.random.rand(len(height_vec)) * 2 - 1) * scaled_height_arr
        return turb_coeff

    def apply_external_disturbances(self):
        #Apply external disturbance force
        if np.random.rand() < self.config["disturbance_frequency"]:
            self.current_disturbance = {"vector" : np.array([np.random.rand(), np.random.rand(), 0]), "remaining_life" : np.random.randint(20, 60)}
            #self.current_disturbance["visual_shape"] = p.createVisualShape()
        if self.current_disturbance is None: return
        p.applyExternalForce(self.robot, linkIndex=-1, forceObj=self.current_disturbance["vector"] * self.config["disturbance_intensity"],
                             posObj=[0, 0, 0], flags=p.LINK_FRAME)
        if self.current_disturbance is not None:
            self.current_disturbance["remaining_life"] -= 1
            if self.current_disturbance["remaining_life"] <= 0:
                #p.removeBody(self.current_disturbance["visual_shape"])
                self.current_disturbance = None

    def get_velocity_target(self):
        velocity_target = None
        if self.config["target_vel_source"] == "still":
            velocity_target = np.zeros(4, dtype=np.float32)
        elif self.config["target_vel_source"] == "rnd":
            velocity_target = self.rnd_target_vel_source() / 3
        elif self.config["target_vel_source"] == "rnd":
            throttle, roll, pitch, yaw = self.joystick_controller.get_joystick_input()[:4]
            velocity_target = -throttle, -roll, -pitch, -yaw
        return velocity_target

    def step(self, ctrl):
        bounded_act = np.tanh(ctrl * self.config["action_scaler"])

        # Take into account motor delay
        self.update_motor_vel(bounded_act)

        # Make turbulence near ground
        #motor_positions_near_ground = p.getLinkStates(self.robot, linkIndices=[1, 3, 5, 7])
        #turbulence_coeffs = self.calc_turbulence_coeffs([pos[0][2] for pos in motor_positions_near_ground])

        # Apply forces
        for i in range(4):
            motor_force_w_noise = np.clip(self.current_motor_velocity_vec[i] * self.motor_power_variance_vector[i]
                                          + self.current_motor_velocity_vec[i], 0, 1)
            motor_force_scaled = motor_force_w_noise * self.robot_params["motor_force_multiplier"]
            p.applyExternalForce(self.robot, linkIndex=i * 2 + 1, forceObj=[0, 0, motor_force_scaled],
                                 posObj=[0, 0, 0], flags=p.LINK_FRAME)
            p.applyExternalTorque(self.robot, linkIndex=i * 2 + 1, torqueObj=[0, 0, self.current_motor_velocity_vec[i] * self.parasitic_torque_dir_vec[i]],
                                  flags=p.LINK_FRAME)

        #self.apply_external_disturbances()

        p.stepSimulation()
        self.step_ctr += 1

        # Read current velocity target
        velocity_target = self.get_velocity_target()

        torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel = self.get_obs()
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        p_vel = np.mean(np.square(np.array([torso_vel[2], torso_vel[1], torso_vel[0], torso_angular_vel[2]])
                            - np.array(velocity_target))) * 0.1
        p_vel = np.clip(p_vel, -3, 3)
        p_pose = np.clip(np.mean(np.square(np.array([roll, pitch]))) * 0.2, -1, 1)
        p_position = np.clip(np.mean(np.square(np.array(torso_pos) - np.array([0,0,1.]))) * .2, -1, 1)
        p_rot = np.mean(np.square(torso_angular_vel[2])) * 0.1
        p_rot = np.clip(p_rot, -1, 1)
        r = 1 - p_position - p_pose - p_rot

        done = (self.step_ctr > self.config["max_steps"]) \
               or np.any(np.array([roll, pitch]) > np.pi / 2) \
               or (abs(np.array(torso_pos) - np.array([0,0,1.])) > np.array([1.5,1.5,0.7])).any()

        obs = np.concatenate((torso_pos, torso_quat, torso_vel, torso_angular_vel)).astype(np.float32)

        return obs, r, done, {}

    def reset(self, force_randomize=None):
        if self.config["randomize_env"]:
            self.robot = self.load_robot()

        self.step_ctr = 0
        self.current_disturbance = None
        self.motor_power_variance_vector = np.ones(4) - np.random.rand(4) * self.config["motor_power_variance"]

        p.resetJointState(self.robot, 0, targetValue=0, targetVelocity=0)
        p.resetBasePositionAndOrientation(self.robot, self.config["starting_pos"], [0, 0, 0, 1], physicsClientId=self.client_ID)
        obs, _, _, _ = self.step(np.zeros(self.act_dim))
        return obs

    def demo(self):
        for i in range(100):
            self.reset()
            act = np.array([0.4, 0.4, 0.4, 0.4])

            for i in range(self.config["max_steps"]):
                obs, r, done, _ = self.step(act)
                time.sleep(0.01)

            self.reset()

    def kill(self):
        p.disconnect()

    def close(self):
        self.kill()

if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = QuadrotorBulletEnv(env_config)
    env.demo()