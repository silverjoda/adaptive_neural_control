import os
import time
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import torch as T
import random

import src.my_utils as my_utils
from src.envs.bullet_quadrotor.peripherals import *
from src.envs.bullet_quadrotor.pid_controller import *

class QuadrotorBulletEnv(gym.Env):
    def __init__(self, config):
        """
        :param config: dict
        """
        if self.seed is not None:
            # Set random seed from config
            self.seed(config["seed"])
        else:
            # Set seed randomly from current time
            rnd_seed = int((time.time() % 1) * 10000000)
            np.random.seed(rnd_seed)
            T.manual_seed(rnd_seed + 1)

        self.config = config

        # (x_delta, y_delta, z_delta), (qx,qy,qz,qw), (x_vel,y_vel,z_vel), (x_ang_vel, y_ang_vel, z_ang_vel)
        self.raw_obs_dim = 13

        # Specify what you want your observation to consist of (how many past observations of each type)
        self.obs_dim = self.config["obs_input"] * self.raw_obs_dim \
                       + self.config["act_input"] * 4 \
                       + self.config["rew_input"] * 1 \
                       + self.config["latent_input"] * 7 \
                       + self.config["step_counter"] * 1
        self.act_dim = 4
        self.reactive_torque_dir_vec = [1, -1, -1, 1]

        # Episode variables
        self.step_ctr = 0
        self.episode_ctr = 0

        # Velocity [0,1] which the motors are currently turning at (unobservable)
        self.current_motor_velocity_vec = np.array([0.,0.,0.,0.])

        if (config["animate"]):
          self.client_ID = p.connect(p.GUI)
        else:
          self.client_ID = p.connect(p.DIRECT)

        # Set simulation parameters
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setTimeStep(self.config["sim_timestep"], physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        # Instantiate instances of joystick controller (human input) and pid controller
        self.joystick_controller = JoyController(self.config)
        self.pid_controller = PIDController(self.config)

        # Load robot model and floor
        self.robot, self.plane = self.load_robot()

        # Define observation and action space (for gym)
        self.observation_space = spaces.Box(low=-self.config["observation_bnd"], high=self.config["observation_bnd"], shape=(self.obs_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

        # Temporally correlated noise for disturbances
        self.rnd_target_vel_source = my_utils.SimplexNoise(4, 15)

        self.obs_queue = [np.zeros(self.raw_obs_dim, dtype=np.float32) for _ in range(np.maximum(1, self.config["obs_input"]) + self.randomized_params["input_transport_delay"])]
        self.act_queue = [np.zeros(self.act_dim,dtype=np.float32) for _ in range(np.maximum(1, self.config["act_input"]) + self.randomized_params["output_transport_delay"])]
        self.rew_queue = [np.zeros(1, dtype=np.float32) for _ in range(np.maximum(1, self.config["rew_input"]) + self.randomized_params["input_transport_delay"])]

        self.create_targets()

    def seed(self, seed=None):
        self.seed = seed
        np.random.seed(self.seed)
        print("Setting seed")

    def create_targets(self):
        target_visualshape = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                        radius=0.1,
                                                        rgbaColor=[0, 1, 0, 1],

                                                        physicsClientId=self.client_ID)
        self.target_body = p.createMultiBody(baseMass=0,
                                         baseVisualShapeIndex=target_visualshape,
                                         basePosition=self.config["target_pos"],
                                         physicsClientId=self.client_ID)

    def set_randomize_env(self, rnd):
        self.config["randomize_env"] = rnd

    def load_robot(self):
        if not hasattr(self, 'robot'):
            self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]),
                               physicsClientId=self.client_ID)
            self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        # Randomize robot params
        self.randomized_params = {"mass": self.config["default_mass"] + (np.random.rand() * 0.6 - 0.3) * self.config["randomize_env"],
                                 "boom": self.config["default_boom_length"] + (np.random.rand() * 0.2 - 0.05) * self.config["randomize_env"],
                                 "motor_inertia_coeff": self.config["default_motor_inertia_coeff"] + (np.random.rand() * 0.4 - 0.2) * self.config["randomize_env"],
                                 "motor_force_multiplier": self.config["default_motor_force_multiplier"] + (np.random.rand() * 4 - 1.5) * self.config["randomize_env"],
                                 "motor_power_variance_vector": np.ones(4) - np.random.rand(4) * self.config["default_motor_power_variance"] * self.config["randomize_env"],
                                 "input_transport_delay": self.config["input_transport_delay"] + np.random.choice(range(self.config["maximum_random_input_transport_delay"])) * self.config["randomize_env"],
                                 "output_transport_delay": self.config["output_transport_delay"] + np.random.choice(range(self.config["maximum_random_output_transport_delay"])) * self.config["randomize_env"]}

        self.randomized_params_list_norm = []
        self.randomized_params_list_norm.append((self.randomized_params["mass"] - self.config["default_mass"]) * (1. / 0.3))
        self.randomized_params_list_norm.append((self.randomized_params["boom"] - self.config["default_boom_length"] - 0.05) * (1. / 0.2))
        self.randomized_params_list_norm.append((self.randomized_params["motor_inertia_coeff"] - self.config["default_motor_inertia_coeff"]) * (1. / 0.2))
        self.randomized_params_list_norm.append((self.randomized_params["motor_force_multiplier"] - self.config["default_motor_force_multiplier"] - 1.5) * (1. / 4))
        self.randomized_params_list_norm.extend((self.randomized_params["motor_power_variance_vector"] - self.config["default_motor_power_variance"]) * (1. / 0.1))
        self.randomized_params_list_norm.append((self.randomized_params["input_transport_delay"] - self.config["maximum_random_input_transport_delay"] / 2) / self.config["maximum_random_input_transport_delay"])
        self.randomized_params_list_norm.append((self.randomized_params["output_transport_delay"] - self.config["maximum_random_output_transport_delay"] / 2) / self.config["maximum_random_output_transport_delay"])

        # Change base mass
        p.changeDynamics(self.robot, -1, mass=self.randomized_params["mass"], physicsClientId=self.client_ID)

        return self.robot, self.plane

    def get_obs(self):
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)
        torso_euler = my_utils._quat_to_euler(*torso_quat)
        obs = [torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel]
        return obs

    def update_motor_vel(self, ctrl):
        self.current_motor_velocity_vec = np.clip(self.current_motor_velocity_vec * self.randomized_params["motor_inertia_coeff"] +
                                                  np.array(ctrl) * (1 - self.randomized_params["motor_inertia_coeff"]), 0, 1)

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
                                 posObj=[0, 0, 0], flags=p.LINK_FRAME)
        else:
            p.applyExternalTorque(self.robot, linkIndex=-1,
                                 torqueObj=self.current_disturbance["vector"] * self.config["disturbance_intensity"] * 0.15,
                                 flags=p.LINK_FRAME)
        if self.current_disturbance is not None:
            self.current_disturbance["remaining_life"] -= 1
            if self.current_disturbance["remaining_life"] <= 0:
                #p.removeBody(self.current_disturbance["visual_shape"])
                self.current_disturbance = None

    def get_input_target(self):
        velocity_target = None
        if self.config["target_input_source"] == "still":
            velocity_target = np.zeros(4, dtype=np.float32)
        elif self.config["target_input_source"] == "rnd":
            velocity_target = self.rnd_target_vel_source()
        elif self.config["target_input_source"] == "joystick":
            throttle, roll, pitch, yaw = self.joystick_controller.get_joystick_input()[:4]
            velocity_target = [-throttle, -roll, -pitch, -yaw]
        return velocity_target

    def render(self, close=False, mode=None):
        #if self.config["animate"]:
        #    time.sleep(self.config["sim_timestep"])
        pass

    def step(self, ctrl_raw):
        # Add new action to queue
        self.act_queue.append(ctrl_raw)
        self.act_queue.pop(0)

        # Simulate the delayed action
        if self.randomized_params["output_transport_delay"] > 0:
            ctrl_raw_unqueued = self.act_queue[-self.config["act_input"]:]
            ctrl_delayed = self.act_queue[-1 - self.randomized_params["output_transport_delay"]]
        else:
            ctrl_raw_unqueued = self.act_queue
            ctrl_delayed = self.act_queue[-1]

        # Scale the action appropriately (neural network gives [-1,1], pid controller [0,1])
        if self.config["controller_source"] == "nn":
            ctrl_processed = np.clip(ctrl_delayed, -1, 1) * 0.5 + 0.5
        else:
            ctrl_processed = ctrl_delayed

        # Take into account motor delay
        self.update_motor_vel(ctrl_processed)

        # Apply motor forces
        for i in range(4):
            motor_force_w_noise = np.clip(self.current_motor_velocity_vec[i] * self.randomized_params["motor_power_variance_vector"][i], 0, 1)
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

        torso_pos, torso_quat, torso_euler, torso_vel, torso_angular_vel = self.get_obs()
        roll, pitch, yaw = torso_euler

        pos_delta = np.array(torso_pos) - np.array(self.config["target_pos"])

        crashed = (abs(pos_delta) > 6.0).any() or ((torso_pos[2] < 0.3) and (abs(roll) > 2.5 or abs(pitch) > 2.5))

        if self.prev_act is not None:
            action_penalty = np.mean(np.square(np.array(ctrl_raw) - np.array(self.prev_act))) * self.config["pen_act_coeff"]
        else:
            action_penalty = 0

        # Calculate true reward
        pen_position = np.mean(np.square(pos_delta)) * self.config["pen_position_coeff"]
        pen_rpy = np.mean(np.square(np.array(torso_euler))) * self.config["pen_rpy_coeff"]
        pen_rotvel = np.mean(np.square(torso_angular_vel)) * self.config["pen_ang_vel_coeff"]
        r_true = - action_penalty - pen_position - pen_rpy - pen_rotvel

        # Calculate proxy reward (for learning purposes)
        pen_position = np.mean(my_utils.universal_lf(pos_delta, -1, self.config["pen_position_c"]))
        pen_yaw = np.mean(my_utils.universal_lf(yaw, -1, self.config["pen_position_c"]))
        r = - pen_position - pen_yaw

        self.rew_queue.append([r])
        self.rew_queue.pop(0)
        if self.randomized_params["input_transport_delay"] > 0:
            r_unqueued = self.rew_queue[-self.config["rew_input"]:]
        else:
            r_unqueued = self.rew_queue

        done = (self.step_ctr > self.config["max_steps"])

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

        obs_dict = {"pos_delta" : pos_delta,
                    "torso_quat" : torso_quat,
                    "torso_vel" : torso_vel,
                    "torso_angular_vel" : torso_angular_vel,
                    "reward" : r_true,
                    "action" : ctrl_raw}

        self.prev_act = ctrl_raw

        return obs, r, done, {"obs_dict" : obs_dict, "randomized_params" : self.randomized_params, "true_rew" : r_true}

    def reset(self, force_randomize=None):
        # If we are randomizing env every episode, then delete everything and load robot again
        if self.config["randomize_env"]:
            self.robot, self.plane = self.load_robot()

        # Reset PID variables
        self.pid_controller.setup_stabilization_control()

        self.step_ctr = 0
        self.episode_ctr += 1
        self.current_disturbance = None
        self.prev_act = None

        self.obs_queue = [np.zeros(self.raw_obs_dim, dtype=np.float32) for _ in range(
            np.maximum(1, self.config["obs_input"]) + self.randomized_params["input_transport_delay"])]
        self.act_queue = [np.zeros(self.act_dim, dtype=np.float32) for _ in range(
            np.maximum(1, self.config["act_input"]) + self.randomized_params["output_transport_delay"])]
        self.rew_queue = [np.zeros(1, dtype=np.float32) for _ in range(
            np.maximum(1, self.config["rew_input"]) + self.randomized_params["input_transport_delay"])]

        if self.config["rnd_init"]:
            difc = self.config["init_difficulty"]
            rnd_starting_pos_delta = np.random.rand(3) * 3. * difc - 1.5 * difc
            rnd_starting_orientation = p.getQuaternionFromEuler(np.random.rand(3) * 2 * difc - 1 * difc, physicsClientId=self.client_ID)
            rnd_starting_lin_velocity = np.random.rand(3) * 2 * difc - 1 * difc
            rnd_starting_rot_velocity = np.random.rand(3) * 1 * difc - .5 * difc
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

    def demo(self):
        k = 0
        while True:
            self.reset()
            act = np.array([-0.7, -0.7, -0.7, -0.7])

            for i in range(self.config["max_steps"]):
                obs, r, done, _ = self.step(act)

            k += 1

    def demo_joystick_PID(self):
        obs = self.reset()
        while True:
            input_target = self.get_input_target()

            act = self.pid_controller.calculate_stabilization_action(obs[3:7], obs[10:13], input_target)
            obs, r, done, _ = self.step(act)
            if done: obs = self.reset()

    def demo_joystick_NN(self):
        self.config["policy_type"] = "mlp"

        # Load neural network policy
        from stable_baselines import TD3
        src_file = os.path.split(os.path.split(os.path.join(os.path.dirname(os.path.realpath(__file__))))[0])[0]
        try:
            model = TD3.load(os.path.join(src_file, "algos/SB/agents/xxx_SB_policy.zip"))
        except:
            model = None
            print("Failed to load nn. ")

        obs = self.reset()
        while True:
            velocity_target = self.get_input_target()

            if self.config["controller_source"] == "nn":
                if model == None:
                    act = np.random.rand(self.act_dim) * 2 - 1
                else:
                    act, _states = model.predict(obs, deterministic=True)
            else:
                act = self.pid_controller.calculate_stabilization_action(obs[3:7], obs[10:13], velocity_target)
            obs, r, done, _ = self.step(act)
            if done: obs = self.reset()

    def deploy_trained_model(self):
        # Load neural network policy
        from stable_baselines3 import TD3
        src_file = os.path.split(os.path.split(os.path.join(os.path.dirname(os.path.realpath(__file__))))[0])[0]

        try:
            model = TD3.load(os.path.join(src_file, "algos/SB/agents/QUAD_TD3_OPTUNA_policy"))
        except:
            model = None
            print("Failed to load nn. ")

        obs = self.reset()
        while True:
            velocity_target = self.get_input_target()

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

                velocity_target = self.get_input_target()
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

    def generate_urdf_from_specs(self):
        BOOM_LEN = 0.2

        # # # Write params to URDF file
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]), "r") as in_file:
            buf = in_file.readlines()

        index = self.config["urdf_name"].find('.urdf')
        output_urdf = self.config["urdf_name"][:index] + '_generated' + self.config["urdf_name"][index:]

        # Change link lengths in urdf
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_urdf), "w") as out_file:
            for line in buf:
                if "<cylinder radius" in line:
                    out_file.write(f'          <cylinder radius="0.015" length="{BOOM_LEN}"/>\n')
                elif line.rstrip('\n').endswith('<!--boomorigin-->'):
                    out_file.write(
                        f'        <origin xyz="0 {BOOM_LEN / 2.} 0.0" rpy="-1.5708 0 0" /><!--boomorigin-->\n')
                elif line.rstrip('\n').endswith('<!--motorpos-->'):
                    out_file.write(
                        f'      <origin xyz="0 {BOOM_LEN} 0" rpy="0 0 0"/><!--motorpos-->\n')
                else:
                    out_file.write(line)


if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = QuadrotorBulletEnv(env_config)

    #env.demo()
    #env.generate_urdf_from_specs()
    env.demo_joystick_PID()