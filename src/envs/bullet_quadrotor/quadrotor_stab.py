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
import random

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
        else:
            logging.info("No joystick found")
        logging.info("Finished initializing the joystick controller.")
        self.button_x_state = 0

    def get_joystick_input(self):
        pygame.event.pump()
        t_roll, t_pitch, t_yaw, throttle = \
            [self.joystick.get_axis(self.config["joystick_mapping"][i]) for i in range(4)]
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

        return throttle, -t_roll, t_pitch, t_yaw, button_x


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
        self.just_obs_dim = 13
        self.obs_dim = self.config["obs_input"] * self.just_obs_dim \
                       + self.config["act_input"] * 4 \
                       + self.config["rew_input"] * 1 \
                       + self.config["latent_input"] * 7 \
                       + self.config["step_counter"] * 1
        self.act_dim = 4
        self.reactive_torque_dir_vec = [1, -1, -1, 1]

        # Episode variables
        self.step_ctr = 0
        self.current_motor_velocity_vec = np.array([0.,0.,0.,0.])

        if (config["animate"]):
          self.client_ID = p.connect(p.GUI)
        else:
          self.client_ID = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setTimeStep(config["sim_timestep"], physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.robot = self.load_robot()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.obs_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

        self.rnd_target_vel_source = my_utils.SimplexNoise(4, 15)
        self.joystick_controller = JoyController(self.config)

        self.obs_queue = [np.zeros(self.just_obs_dim,dtype=np.float32) for _ in range(np.maximum(1, self.config["obs_input"]) + self.randomized_params["input_transport_delay"])]
        self.act_queue = [np.zeros(self.act_dim,dtype=np.float32) for _ in range(np.maximum(1, self.config["act_input"]) + self.randomized_params["output_transport_delay"])]
        self.rew_queue = [np.zeros(1, dtype=np.float32) for _ in range(np.maximum(1, self.config["rew_input"]) + self.randomized_params["input_transport_delay"])]

        self.setup_stabilization_control()

    def seed(self, seed=None):
        self.seed = seed
        np.random.seed(self.seed)
        T.manual_seed(self.seed)
        print("Setting seed")

    def setup_stabilization_control(self):
        self.e_roll_prev = 0
        self.e_pitch_prev = 0
        self.e_yaw_prev = 0

        self.e_roll_accum = 0
        self.e_pitch_accum = 0

    def set_randomize_env(self, rnd):
        self.config["randomize_env"] = rnd

    def load_robot(self):
        if hasattr(self, 'robot'):
            pass
            #p.removeBody(self.robot)
        else:
            pass
            self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]),
                               physicsClientId=self.client_ID)

        # Randomize robot params
        self.randomized_params = {"mass": 0.8 + (np.random.rand() * 0.6 - 0.3) * self.config["randomize_env"],
                                 "boom": 0.15 + (np.random.rand() * 0.3 - 0.1) * self.config["randomize_env"],
                                 "motor_alpha": 0.1 + (np.random.rand() * 0.04 - 0.02) * self.config["randomize_env"],
                                 "motor_force_multiplier": 8 + (np.random.rand() * 4 - 1.5) * self.config["randomize_env"],
                                 "motor_power_variance_vector": np.ones(4) - np.random.rand(4) * 0.10 * self.config["randomize_env"],
                                 "input_transport_delay": self.config["input_transport_delay"] + 1 * np.random.choice([0,1,2], p=[0.4, 0.5, 0.1]) * self.config["randomize_env"],
                                 "output_transport_delay": self.config["output_transport_delay"] + 1 * np.random.choice([0,1,2], p=[0.4, 0.5, 0.1]) * self.config["randomize_env"]}

        self.randomized_params_list_norm = []
        self.randomized_params_list_norm.append((self.randomized_params["mass"] - 0.7) * (1. / 0.3))
        self.randomized_params_list_norm.append((self.randomized_params["motor_alpha"] - 0.05) * (1. / 0.02))
        self.randomized_params_list_norm.append((self.randomized_params["motor_force_multiplier"] - 8) * (1. / 1.5))
        self.randomized_params_list_norm.extend((self.randomized_params["motor_power_variance_vector"] - 0.95) * (1. / 0.05))
        self.randomized_params_list_norm.append(self.randomized_params["input_transport_delay"] - 1)
        self.randomized_params_list_norm.append(self.randomized_params["output_transport_delay"] - 1)

        # # # Write params to URDF file
        # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]), "r") as in_file:
        #     buf = in_file.readlines()
        #
        # index = self.config["urdf_name"].find('.urdf')
        # output_urdf = self.config["urdf_name"][:index] + '_rnd' + self.config["urdf_name"][index:]
        #
        # # Change link lengths in urdf
        # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_urdf), "w") as out_file:
        #     for line in buf:
        #         if "<cylinder radius" in line:
        #             out_file.write(f'          <cylinder radius="0.015" length="{self.randomized_params["boom"]}"/>\n')
        #         elif line.rstrip('\n').endswith('<!--boomorigin-->'):
        #             out_file.write(f'        <origin xyz="0 {self.randomized_params["boom"] / 2.} 0.0" rpy="-1.5708 0 0" /><!--boomorigin-->\n')
        #         elif line.rstrip('\n').endswith('<!--motorpos-->'):
        #             out_file.write(f'      <origin xyz="0 {self.randomized_params["boom"]} 0" rpy="0 0 0"/><!--motorpos-->\n')
        #         else:
        #             out_file.write(line)
        #
        # # Load urdf
        # self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_urdf), physicsClientId=self.client_ID)

        # Change base mass
        p.changeDynamics(self.robot, -1, mass=self.randomized_params["mass"], physicsClientId=self.client_ID)

        return self.robot

    def get_obs(self):
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)
        torso_euler = my_utils._quat_to_euler(*torso_quat)
        obs = [torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel]
        return obs

    def update_motor_vel(self, ctrl):
        #self.current_motor_velocity_vec = np.clip(self.current_motor_velocity_vec * self.randomized_params["motor_inertia_coeff"] +
         #                                         np.array(ctrl) * (1 - self.randomized_params["motor_inertia_coeff"]), 0, 1)
        ctrl_clipped = np.clip(ctrl, 0, 1)
        bot = np.minimum(self.current_motor_velocity_vec, ctrl_clipped)
        top = np.maximum(self.current_motor_velocity_vec, ctrl_clipped)
        raw = self.current_motor_velocity_vec + self.randomized_params["motor_alpha"] * np.sign(ctrl - self.current_motor_velocity_vec)
        self.current_motor_velocity_vec = np.clip(raw, bot, top)


    def render(self, close=False, mode=None):
        if self.config["animate"]:
            time.sleep(self.config["sim_timestep"])

    def apply_external_disturbances(self):
        #Apply external disturbance force
        if np.random.rand() < self.config["disturbance_frequency"]:
            self.current_disturbance = {"vector" : np.array([2 * np.random.rand() - 1.0, 2 * np.random.rand() - 1.0, 0.4 * np.random.rand() - 0.2]),
                                        "remaining_life" : np.random.randint(10, 40),
                                        "effect" : np.random.choice(["translation", "rotation"])}
            #self.current_disturbance["visual_shape"] = p.createVisualShape()
        if self.current_disturbance is None: return
        if self.current_disturbance["effect"] == "translation":
            p.applyExternalForce(self.robot, linkIndex=-1, forceObj=self.current_disturbance["vector"] * self.config["disturbance_intensity"],
                                 posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.client_ID)
        else:
            p.applyExternalTorque(self.robot, linkIndex=-1,
                                 torqueObj=self.current_disturbance["vector"] * self.config["disturbance_intensity"] * 0.15,
                                 flags=p.LINK_FRAME, physicsClientId=self.client_ID)
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
            velocity_target = self.rnd_target_vel_source()
        elif self.config["target_vel_source"] == "joystick":
            throttle, roll, pitch, yaw = self.joystick_controller.get_joystick_input()[:4]
            velocity_target = [-throttle, -roll, -pitch, -yaw]
        return velocity_target

    def calculate_stabilization_action(self, orientation, angular_velocities, targets):
        roll, pitch, _ = p.getEulerFromQuaternion(orientation, physicsClientId=self.client_ID)
        roll_vel, pitch_vel, yaw_vel = angular_velocities
        t_throttle, t_roll, t_pitch, t_yaw_vel = targets

        # Increase t_yaw_vel because it's slow as shit
        t_yaw_vel *= 5

        #print(f"Throttle_target: {t_throttle}, Roll_target: {t_roll}, Pitch_target: {t_pitch}, Yaw_vel_target: {t_yaw_vel}")
        #print(f"Roll: {roll}, Pitch: {pitch}, Yaw_vel: {yaw_vel}")

        # Target errors
        e_roll = t_roll - roll
        e_pitch = t_pitch - pitch
        e_yaw = t_yaw_vel - yaw_vel

        decay_fac = 0.7
        self.e_roll_accum = self.e_roll_accum * decay_fac + e_roll
        self.e_pitch_accum = self.e_pitch_accum * decay_fac + e_pitch

        # Desired correction action
        roll_act = e_roll * self.config["p_roll"]  + (e_roll - self.e_roll_prev) * self.config["d_roll"] + self.e_roll_accum * self.config["i_roll"]
        pitch_act = e_pitch * self.config["p_pitch"] + (e_pitch - self.e_pitch_prev) * self.config["d_pitch"] + self.e_pitch_accum * self.config["i_pitch"]
        yaw_act = e_yaw * self.config["p_yaw"] + (e_yaw - self.e_yaw_prev) * self.config["d_yaw"]

        self.e_roll_prev = e_roll
        self.e_pitch_prev = e_pitch
        self.e_yaw_prev = e_yaw

        m_1_act_total = + roll_act - pitch_act + yaw_act
        m_2_act_total = - roll_act - pitch_act - yaw_act
        m_3_act_total = + roll_act + pitch_act - yaw_act
        m_4_act_total = - roll_act + pitch_act + yaw_act

        # Translate desired correction actions to servo commands
        m_1 = np.clip(t_throttle + m_1_act_total, 0, 1)
        m_2 = np.clip(t_throttle + m_2_act_total, 0, 1)
        m_3 = np.clip(t_throttle + m_3_act_total, 0, 1)
        m_4 = np.clip(t_throttle + m_4_act_total, 0, 1)

        #print([m_1, m_2, m_3, m_4])

        if np.max([m_1, m_2, m_3, m_4]) > 1.1:
            print("Warning: motor commands exceed 1.0. This signifies an error in the system", m_1, m_2, m_3, m_4, t_throttle)

        return m_1, m_2, m_3, m_4

    def step(self, ctrl_raw):
        if self.prev_act is not None:
            raw_act_smoothness_pen = np.mean(np.square(np.array(ctrl_raw) - np.array(self.prev_act))) * 0.01
        else:
            raw_act_smoothness_pen = 0
        self.prev_act = ctrl_raw

        self.act_queue.append(ctrl_raw)
        self.act_queue.pop(0)
        if self.randomized_params["output_transport_delay"] > 0:
            ctrl_raw_unqueued = self.act_queue[-self.config["act_input"]:]
            ctrl_delayed = self.act_queue[-1 -self.randomized_params["output_transport_delay"]]
        else:
            ctrl_raw_unqueued = self.act_queue
            ctrl_delayed = self.act_queue[-1]

        if self.config["controller_source"] == "nn":
            ctrl_processed = np.clip(ctrl_delayed * self.config["action_scaler"], -1, 1) * 0.5 + 0.5
        else:
            ctrl_processed = ctrl_delayed

        # Take into account motor delay
        self.update_motor_vel(ctrl_processed)

        # Apply forces
        for i in range(4):
            motor_force_w_noise = np.clip(self.current_motor_velocity_vec[i] * self.randomized_params["motor_power_variance_vector"][i]
                                          + self.current_motor_velocity_vec[i], 0, 1)
            motor_force_scaled = motor_force_w_noise * self.randomized_params["motor_force_multiplier"]
            p.applyExternalForce(self.robot,
                                 linkIndex=i * 2 + 1,
                                 forceObj=[0, 0, motor_force_scaled],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME, physicsClientId=self.client_ID)
            p.applyExternalTorque(self.robot,
                                  linkIndex=i * 2 + 1,
                                  torqueObj=[0, 0, self.current_motor_velocity_vec[i] * self.reactive_torque_dir_vec[i] * self.config["propeller_parasitic_torque_coeff"]],
                                  flags=p.LINK_FRAME, physicsClientId=self.client_ID)

        self.apply_external_disturbances()

        p.stepSimulation(physicsClientId=self.client_ID)
        if self.config["animate"]:
            time.sleep(self.config["sim_timestep"])
        self.step_ctr += 1

        # Read current velocity target
        velocity_target = self.get_velocity_target()

        torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel = self.get_obs()
        roll, pitch, yaw = torso_euler
        pos_delta = np.array(torso_pos) - np.array(self.config["target_pos"])

        p_position = np.mean(np.abs(pos_delta)) * 1.0
        p_rp = np.clip(np.mean(np.abs(np.array([yaw]))) * 1.0, -3, 3)
        #p_rotvel = np.clip(np.mean(np.square(torso_angular_vel[2])) * 0.1, -1, 1)
        r = 1.0 - p_position - p_rp - raw_act_smoothness_pen

        self.rew_queue.append([r])
        self.rew_queue.pop(0)
        if self.randomized_params["input_transport_delay"] > 0:
            r_unqueued = self.rew_queue[-self.config["rew_input"]:]
        else:
            r_unqueued = self.rew_queue

        if torso_pos[2] < 0.3:
            velocity_target[0] = 0.3 - torso_pos[2]

        done = (self.step_ctr > self.config["max_steps"]) or (abs(pos_delta) > 1.7).any() or abs(roll) > 2. or abs(pitch) > 2.

        compiled_obs = pos_delta, torso_quat, torso_vel, torso_angular_vel
        compiled_obs_flat = [item for sublist in compiled_obs for item in sublist]
        self.obs_queue.append(compiled_obs_flat)
        self.obs_queue.pop(0)

        if self.randomized_params["input_transport_delay"] > 0:
            obs_raw_unqueued = self.obs_queue[-self.config["obs_input"]:]
        else:
            obs_raw_unqueued = self.obs_queue

        aux_obs = []
        for i in range(len(obs_raw_unqueued)):
            t_obs = []
            t_obs.extend(obs_raw_unqueued[i])
            if self.config["act_input"] > 0:
                t_obs.extend(ctrl_raw_unqueued[i])
            if self.config["rew_input"] > 0:
                t_obs.extend(r_unqueued[i])
            if self.config["step_counter"] > 0:
                def step_ctr_to_obs(step_ctr):
                    return (float(step_ctr) / self.config["max_steps"]) * 2 - 1

                t_obs.extend([step_ctr_to_obs(np.maximum(self.step_ctr - i - 1, 0))])
            aux_obs.extend(t_obs)

        if self.config["latent_input"]:
            aux_obs.extend(self.randomized_params_list_norm)

        obs = np.array(aux_obs).astype(np.float32)

        return obs, r, done, {"aux_obs" : aux_obs, "randomized_params" : self.randomized_params}

    def reset(self, force_randomize=None):
        if self.config["randomize_env"]:
            self.robot = self.load_robot()

        self.setup_stabilization_control()

        self.step_ctr = 0
        self.current_disturbance = None
        self.prev_act = None

        if self.config["rnd_init"]:
            rnd_starting_pos_delta = np.random.rand(3) * 1. - .5
            rnd_starting_orientation = p.getQuaternionFromEuler([np.random.rand(1) * .4 - 0.2, np.random.rand(1) * .4 - 0.2, np.random.rand(1) * 1 - .5], physicsClientId=self.client_ID)
            rnd_starting_lin_velocity = np.random.rand(3) * .4 - .2
            rnd_starting_rot_velocity = np.random.rand(3) * .2 - .1
        else:
            rnd_starting_pos_delta = np.zeros(3)
            rnd_starting_orientation = np.array([0,0,0,1])
            rnd_starting_lin_velocity = np.zeros(3)
            rnd_starting_rot_velocity = np.zeros(3)

        p.resetJointState(self.robot, 0, targetValue=0, targetVelocity=0, physicsClientId=self.client_ID)
        p.resetBasePositionAndOrientation(self.robot, self.config["starting_pos"] + rnd_starting_pos_delta, rnd_starting_orientation, physicsClientId=self.client_ID)
        p.resetBaseVelocity(self.robot,linearVelocity=rnd_starting_lin_velocity, angularVelocity=rnd_starting_rot_velocity, physicsClientId=self.client_ID)
        obs, _, _, _ = self.step(np.zeros(self.act_dim) + 0.1)
        return obs

    def get_original_reward(self):
        return 0

    def demo(self):
        for i in range(100):
            self.reset()
            act = np.array([-0.7, -0.7, -0.7, -0.7])

            for i in range(self.config["max_steps"]):
                obs, r, done, _ = self.step(act)
                time.sleep(self.config["sim_timestep"])

            self.reset()

    def demo_joystick(self):
        self.config["policy_type"] = "mlp"

        # Load neural network policy
        from stable_baselines import A2C
        src_file = os.path.split(os.path.split(os.path.join(os.path.dirname(os.path.realpath(__file__))))[0])[0]
        try:
            model = A2C.load(os.path.join(src_file, "algos/SB/agents/xxx_SB_policy.zip"))
        except:
            model = None
            print("Failed to load nn. ")

        obs = self.reset()
        while True:
            velocity_target = self.get_velocity_target()
            #print(velocity_target)

            if self.config["controller_source"] == "nn":
                if model == None:
                    act = np.random.rand(self.act_dim) * 2 - 1
                else:
                    act, _states = model.predict(obs, deterministic=True)
            else:
                act = self.calculate_stabilization_action(obs[3:7], obs[10:13], velocity_target)
            obs, r, done, _ = self.step(act)
            if done: obs = self.reset()

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

                velocity_target = self.get_velocity_target()
                #print(velocity_target)

                if self.config["controller_source"] == "pid":
                    # Read target control inputs
                    m_1, m_2, m_3, m_4 = self.calculate_stabilization_action(rotation_rob, angular_vel_rob, velocity_target)
                else:
                    m_1, m_2, m_3, m_4 = policy(obs)

                data_position.append(position_rob)
                data_vel.append(vel_rob)
                data_rotation.append(rotation_rob)
                data_angular_vel.append(angular_vel_rob)
                data_timestamp.append(0)
                data_action.append([m_1, m_2])

                # Write control to servos
                obs = self.step([m_1, m_2, m_3, m_4])

                # Sleep to maintain correct FPS
                while time.time() - iteration_starttime < self.config["sim_timestep"]: pass
        except KeyboardInterrupt:
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

    def kill(self):
        p.disconnect(physicsClientId=self.client_ID)

    def close(self):
        self.kill()


if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = QuadrotorBulletEnv(env_config)
    env.demo_joystick()
    #env.gather_data()
