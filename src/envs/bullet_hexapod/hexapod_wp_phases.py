import math
import os
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
import torch as T
from gym import spaces
from opensimplex import OpenSimplex

# INFO: To mirror quaternion along x-z plane (or y axis) just use q_mirror = [qx, -qy, qz, -qw]

class HexapodBulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, config):
        self.config = config
        self.seed = self.config["seed"]
        
        if self.seed is not None:
            self.set_seed(self.seed, self.seed)
        else:
            rnd_seed = int((time.time() % 1) * 10000000)
            self.set_seed(rnd_seed, rnd_seed + 1)

        if (self.config["animate"]):
            self.client_ID = p.connect(p.GUI)
            print(" --Starting GUI mode-- ")
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        # Environment parameters
        self.act_dim = 18 * 2
        self.just_obs_dim = 57
        self.obs_dim = self.config["obs_input"] * self.just_obs_dim \
                       + self.config["act_input"] * self.act_dim \
                       + self.config["rew_input"] * 1 \
                       + self.config["latent_input"] * 6 \
                       + self.config["step_counter"] * 1 \

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        self.joints_rads_low = np.array(config["joints_rads_low"] * 6)
        self.joints_rads_high = np.array(config["joints_rads_high"] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.coxa_joint_ids = range(0, 18, 3)
        self.femur_joint_ids = range(1, 18, 3)
        self.tibia_joint_ids = range(2, 18, 3)
        self.left_joints_ids = [0, 1, 2, 6, 7, 8, 12, 13, 14]
        self.right_joints_ids = [3, 4, 5, 9, 10, 11, 15, 16, 17]

        self.time_vector = np.zeros(18)
        self.time_tick = 0.4
        self.bounds_tick = 0.04
        self.time_tick_action_scalar = 0.3
        self.phase_amplitude_mult = .8

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.urdf_name = config["urdf_name"]
        self.robot = self.load_robot()

        if config["terrain_name"] == "flat":
            self.terrain = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)
        else:
            self.terrain = self.generate_rnd_env()

        # Episodal parameters
        self.step_ctr = 0
        self.episode_ctr = 0
        self.episode_rew_list = []
        self.rew_mean = 0.
        self.rew_std = 1.
        self.xd_queue = []
        self.joint_work_done_arr_list = []
        self.joint_angle_arr_list = []
        self.prev_yaw_dev = 0
        self.max_dist_travelled = 0
        self.target_vel_nn_input = 0

        self.create_targets()

        self.obs_queue = [np.zeros(self.just_obs_dim,dtype=np.float32) for _ in range(np.maximum(1, self.config["obs_input"]))]
        self.act_queue = [np.zeros(self.act_dim,dtype=np.float32) for _ in range(np.maximum(1, self.config["act_input"]))]
        self.rew_queue = [np.zeros(1,dtype=np.float32) for _ in range(np.maximum(1, self.config["rew_input"]))]


    def set_seed(self, np_seed, T_seed):
        np.random.seed(np_seed)
        T.manual_seed(T_seed)


    def create_targets(self):
        self.target = None
        self.target_visualshape = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                        radius=0.1,
                                                        rgbaColor=[1, 0, 0, 1],
                                                        physicsClientId=self.client_ID)
        self.update_targets()

    def update_targets(self):
        if self.target is None:
            self.target = np.array(self.config["target_spawn_mu"]) + np.random.rand(2) * np.array(self.config["target_spawn_sigma"]) - np.array(self.config[
                "target_spawn_sigma"]) / 2

            self.target_body = p.createMultiBody(baseMass=0,
                                                   baseVisualShapeIndex=self.target_visualshape,
                                                   basePosition=[self.target[0], self.target[1], 0],
                                                   physicsClientId=self.client_ID)
        else:
            self.target = np.array(self.config["target_spawn_mu"]) + np.random.rand(2) * np.array(
                self.config["target_spawn_sigma"]) - np.array(self.config["target_spawn_sigma"]) / 2
            p.resetBasePositionAndOrientation(self.target_body, [self.target[0], self.target[1], 0], [0, 0, 0, 1], physicsClientId=self.client_ID)

    def set_randomize_env(self, rnd):
        self.config["randomize_env"] = rnd

    def generate_rnd_env(self):
        if self.config["terrain_name"] is None:
            self.terrain_hm = np.zeros((self.config["env_length"], self.config["env_width"]))
            return self.make_heightfield(self.terrain_hm)

        self.terrain_hm, _ = self.generate_heightmap(self.config["terrain_name"])
        self.terrain_hm /= 255.
        return self.make_heightfield(self.terrain_hm)

    def generate_heightmap(self, env_name):
        current_height = 0
        if env_name == "flat" or env_name is None:
            hm = np.ones((self.config["env_length"], self.config["env_width"])) * current_height

        if env_name == "tiles":
            sf = 3
            hm = np.random.randint(0, 20 * self.config["training_difficulty"], # 15
                                   size=(self.config["env_length"] // sf, self.config["env_width"] // sf)).repeat(sf, axis=0).repeat(sf, axis=1)
            hm_pad = np.zeros((self.config["env_length"], self.config["env_width"]))
            hm_pad[:hm.shape[0], :hm.shape[1]] = hm
            hm = hm_pad + current_height

        if env_name == "pipe":
            pipe_form = np.square(np.linspace(-1.2, 1.2, self.config["env_width"]))
            pipe_form = np.clip(pipe_form, 0, 1)
            hm = 255 * np.ones((self.config["env_width"], self.config["env_length"])) * pipe_form[np.newaxis, :].T
            hm += current_height

        if env_name == "stairs_up":
            hm = np.ones((self.config["env_length"], self.config["env_width"])) * current_height
            stair_height = 14 * self.config["training_difficulty"]
            stair_width = 8

            initial_offset = self.config["env_length"] // 2 - self.config["env_length"] // 14
            n_steps = min(math.floor(self.config["env_length"] / stair_width) - 1, 10)

            for i in range(n_steps):
                hm[initial_offset + i * stair_width: initial_offset + i * stair_width + stair_width, :] = current_height
                current_height += stair_height

            hm[n_steps * stair_width + initial_offset:, :] = current_height

        if env_name == "ramp_up":
            max_height = 200
            hm = np.ones((self.config["env_length"], self.config["env_width"])) * current_height
            initial_offset = self.config["env_length"] // 2 - self.config["env_length"] // 8
            hm[initial_offset:initial_offset + self.config["env_length"] // 4, :] = np.tile(np.linspace(current_height, current_height + max_height, self.config["env_length"] // 4)[:, np.newaxis], (1, self.config["env_width"]))
            current_height += max_height
            hm[initial_offset + self.config["env_length"] // 4:, :] = current_height

        if env_name == "stairs_down":
            stair_height = 14 * self.config["training_difficulty"]
            stair_width = 8

            initial_offset = self.config["env_length"] // 2
            n_steps = min(math.floor(self.config["env_length"] / stair_width) - 1, 10)

            current_height = n_steps * stair_height

            hm = np.ones((self.config["env_length"], self.config["env_width"])) * current_height

            for i in range(n_steps):
                hm[initial_offset + i * stair_width: initial_offset + i * stair_width + stair_width, :] = current_height
                current_height -= stair_height

            hm[n_steps * stair_width + initial_offset:, :] = current_height

        if env_name == "verts":
            wdiv = 4
            ldiv = 14
            hm = np.random.randint(0, 75,
                                   size=(self.config["env_width"] // wdiv, self.config["env_length"] // ldiv),
                                   dtype=np.uint8).repeat(wdiv, axis=0).repeat(ldiv, axis=1)
            hm[:, :50] = 0
            hm[hm < 50] = 0
            hm = 75 - hm

        if env_name == "triangles":
            cw = 10
            # Make even dimensions
            M = math.ceil(self.config["env_width"])
            N = math.ceil(self.config["env_length"])
            hm = np.zeros((M, N), dtype=np.float32)
            M_2 = math.ceil(M / 2)

            # Amount of 'tiles'
            Mt = 2
            Nt = int(self.config["env_length"] / 10.)
            obstacle_height = 50
            grad_mat = np.linspace(0, 1, cw)[:, np.newaxis].repeat(cw, 1)
            template_1 = np.ones((cw, cw)) * grad_mat * grad_mat.T * obstacle_height
            template_2 = np.ones((cw, cw)) * grad_mat * obstacle_height

            for i in range(Nt):
                if np.random.choice([True, False]):
                    hm[M_2 - cw: M_2, i * cw: i * cw + cw] = np.rot90(template_1, np.random.randint(0, 4))
                else:
                    hm[M_2 - cw: M_2, i * cw: i * cw + cw] = np.rot90(template_2, np.random.randint(0, 4))

                if np.random.choice([True, False]):
                    hm[M_2:M_2 + cw:, i * cw: i * cw + cw] = np.rot90(template_1, np.random.randint(0, 4))
                else:
                    hm[M_2:M_2 + cw:, i * cw: i * cw + cw] = np.rot90(template_2, np.random.randint(0, 4))

            hm += current_height

        if env_name == "perlin":
            oSim = OpenSimplex(seed=int(time.time()))

            height = self.config["perlin_height"] * self.config["training_difficulty"] # 30-40

            M = math.ceil(self.config["env_width"])
            N = math.ceil(self.config["env_length"])
            hm = np.zeros((N, M), dtype=np.float32)

            scale_x = 15
            scale_y = 15
            octaves = 4  # np.random.randint(1, 5)
            persistence = 1
            lacunarity = 2

            for i in range(N):
                for j in range(M):
                    for o in range(octaves):
                        sx = scale_x * (1 / (lacunarity ** o))
                        sy = scale_y * (1 / (lacunarity ** o))
                        amp = persistence ** o
                        hm[i][j] += oSim.noise2d(i / sx, j / sy) * amp

            wmin, wmax = hm.min(), hm.max()
            hm = (hm - wmin) / (wmax - wmin) * height
            hm += current_height

        return hm, current_height

    def make_heightfield(self, height_map=None):
        if self.config["terrain_name"] == "flat":
            return
        if hasattr(self, 'terrain'):
            p.removeBody(self.terrain, physicsClientId=self.client_ID)
        if height_map is None:
            heightfieldData = np.zeros(self.config["env_width"] * self.config["max_steps"])
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[self.config["mesh_scale_lat"] , self.config["mesh_scale_lat"] , self.config["mesh_scale_vert"]],
                                                  heightfieldTextureScaling=(self.config["env_width"] - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=self.config["max_steps"],
                                                  numHeightfieldColumns=self.config["env_width"], physicsClientId=self.client_ID)
        else:
            heightfieldData = height_map.ravel(order='F')
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[self.config["mesh_scale_lat"], self.config["mesh_scale_lat"], self.config["mesh_scale_vert"]],
                                                  heightfieldTextureScaling=(self.config["env_width"] - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=height_map.shape[0],
                                                  numHeightfieldColumns=height_map.shape[1],
                                                  physicsClientId=self.client_ID)
        terrain = p.createMultiBody(0, terrainShape, physicsClientId=self.client_ID)

        p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_ID)
        return terrain

    def get_obs(self):
        # Torso
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID) # xyz and quat: x,y,z,w
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)

        contacts = [int(len(p.getContactPoints(self.robot, self.terrain, i * 3 + 2, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1 for i in range(6)]
        ctct_torso = int(len(p.getContactPoints(self.robot, self.terrain, -1, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        #contacts = np.zeros(6)

        # Joints
        obs = p.getJointStates(self.robot, range(18), physicsClientId=self.client_ID) # pos, vel, reaction(6), prev_torque
        joint_angles = []
        joint_velocities = []
        joint_torques = []
        for o in obs:
            joint_angles.append(o[0])
            joint_velocities.append(o[1])
            joint_torques.append(o[3])

        return torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso

    def rads_to_norm(self, joints):
        sjoints = np.array(joints)
        sjoints = ((sjoints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        return sjoints

    def norm_to_rads(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    def render(self, close=False, mode=None):
        if self.config["animate"]:
            time.sleep(self.config["sim_timestep"])

    def load_robot(self):
        # Remove old robot
        if not hasattr(self, 'robot'):
            self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.urdf_name), physicsClientId=self.client_ID)

        # Randomize robot params
        self.randomized_params = {"mass": 1.5 + (np.random.rand() * 1.4 - 0.7) * self.config[
                                "randomize_env"],
                                "lateral_friction": 1.2 + (np.random.rand() * 1.2 - 0.6) * self.config[
                                    "randomize_env"],
                                "max_joint_force": 1.5 + (np.random.rand() * 1.0 - 0.5) * self.config[
                                    "randomize_env"],
                                "actuator_position_gain": 0.3 + (np.random.rand() * 0.4 - 0.2) * self.config[
                                      "randomize_env"],
                                "actuator_velocity_gain": 0.3 + (np.random.rand() * 0.4 - 0.2) * self.config[
                                      "randomize_env"],
                                "max_actuator_velocity": 4.0 + (np.random.rand() * 4.0 - 2.0) * self.config[
                                      "randomize_env"],
                                }

        self.randomized_params_list_norm = []
        self.randomized_params_list_norm.append(
            (self.randomized_params["mass"] - 1.5) * (1. / 0.7))
        self.randomized_params_list_norm.append(
            (self.randomized_params["lateral_friction"] - 1.2) * (1. / 0.6))
        self.randomized_params_list_norm.append(
            (self.randomized_params["max_joint_force"] - 1.0) * (1. / 0.5))
        self.randomized_params_list_norm.append(
            (self.randomized_params["actuator_position_gain"] - 0.3) * (1. / 0.2))
        self.randomized_params_list_norm.append(
            (self.randomized_params["actuator_velocity_gain"] - 0.3) * (1. / 0.2))
        self.randomized_params_list_norm.append(
            (self.randomized_params["max_actuator_velocity"] - 4.0) * (1. / 2.0))

        p.changeDynamics(self.robot, -1, mass=self.randomized_params["mass"],
                         physicsClientId=self.client_ID)
        p.changeDynamics(self.robot, -1, lateralFriction=self.randomized_params["lateral_friction"],
                         physicsClientId=self.client_ID)
        for i in range(6):
            p.changeDynamics(self.robot, 3 * i + 2, lateralFriction=self.randomized_params["lateral_friction"], physicsClientId=self.client_ID)

        return self.robot

    def step(self, ctrl_raw, render=False):
        self.act_queue.append(ctrl_raw)
        self.act_queue.pop(0)

        ctrl_clipped = np.tanh(np.array(ctrl_raw) * self.config["action_scaler"])

        ctrl_joints = ctrl_clipped[:18]
        ctrl_additive = ctrl_clipped[18:]

        #self.joints_rads_low += np.tile(ctrl_bounds[:3], (6)) * self.bounds_tick
        #self.joints_rads_high += np.tile(ctrl_bounds[3:], (6)) * self.bounds_tick
        #self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.time_vector = self.time_vector + ctrl_joints * self.time_tick_action_scalar + self.time_tick
        phases = self.phase_amplitude_mult * np.sin(self.time_vector) + ctrl_additive * 0.5

        scaled_action = self.norm_to_rads(phases)

        for i in range(18):
            p.setJointMotorControl2(bodyUniqueId=self.robot,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=scaled_action[i],
                                    force=self.randomized_params["max_joint_force"],
                                    positionGain=self.randomized_params["actuator_position_gain"],
                                    velocityGain=self.randomized_params["actuator_velocity_gain"],
                                    maxVelocity=self.randomized_params["max_actuator_velocity"],
                                    physicsClientId=self.client_ID)

        # Read out joint angles sequentially (to simulate servo daisy chain delay)
        leg_ctr = 0
        obs_sequential = []
        for i in range(self.config["sim_steps_per_iter"]):
            if leg_ctr < 6:
                obs_sequential.extend(
                    p.getJointStates(self.robot, range(leg_ctr * 3, (leg_ctr + 1) * 3), physicsClientId=self.client_ID))
                leg_ctr += 1
            p.stepSimulation(physicsClientId=self.client_ID)
            if (self.config["animate"] or render) and True: time.sleep(0.00417)

        self.step_ctr += 1

        joint_angles_skewed = []
        for o in obs_sequential:
            joint_angles_skewed.append(o[0])

        # Get all observations
        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso = self.get_obs()
        xd, yd, zd = torso_vel
        thd, phid, psid = torso_angular_vel

        scaled_joint_angles = self.rads_to_norm(joint_angles_skewed)

        # Calculate work done by each motor
        joint_work_done_arr = np.array(joint_torques) * np.array(joint_velocities)
        total_work_pen = np.mean(np.square(joint_work_done_arr))

        # Calculate yaw
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        # Orientation angle
        tar_angle = np.arctan2(self.target[1] - torso_pos[1], self.target[0] - torso_pos[0])
        yaw_deviation = np.min((abs((yaw % 6.283) - (tar_angle % 6.283)), abs(yaw - tar_angle)))

        # Compute heading reward
        yaw_dev_diff = abs(self.prev_yaw_deviation) - abs(yaw_deviation)
        yaw_dev_sign = np.sign(yaw_dev_diff)
        heading_rew = np.minimum(np.abs(yaw_deviation), 3) * np.clip(yaw_dev_sign * np.square(yaw_dev_diff) / (self.config["sim_step"] * self.config["sim_steps_per_iter"]), -2, 2)

        # Check if the agent has reached a target
        target_dist = np.sqrt((torso_pos[0] - self.target[0]) ** 2 + (torso_pos[1] - self.target[1]) ** 2)
        velocity_rew = np.minimum((self.prev_target_dist - target_dist) / (self.config["sim_step"] * self.config["sim_steps_per_iter"]),
                                  self.config["target_vel"]) / self.config["target_vel"]

        if target_dist < self.config["target_proximity_threshold"]:
            reached_target = True
            self.update_targets()
            self.prev_target_dist = np.sqrt(
                (torso_pos[0] - self.target[0]) ** 2 + (torso_pos[1] - self.target[1]) ** 2)
            tar_angle = np.arctan2(self.target[1] - torso_pos[1], self.target[0] - - torso_pos[0])
            yaw_deviation = np.min(
                (abs((yaw % 6.283) - (tar_angle % 6.283)), abs(yaw - tar_angle)))
            self.prev_yaw_deviation = yaw_deviation
        else:
            reached_target = False
            self.prev_target_dist = target_dist
            self.prev_yaw_deviation = yaw_deviation

        r_neg = {"inclination": np.sqrt(np.square(pitch) + np.square(roll)) * self.config["inclination_pen"],
                 "bobbing": np.sqrt(np.square(zd)) * 1.0,
                 "yaw_pen": np.square(tar_angle - yaw) * 0.0}

        r_pos = {"velocity_rew": np.clip(velocity_rew / (1 + abs(yaw_deviation) * 3), -2, 2),
                 "heading_rew" : np.clip(heading_rew * 0.0, -1, 1)}

        r_pos_sum = sum(r_pos.values())
        r_neg_sum = np.maximum(np.minimum(sum(r_neg.values()) * (self.step_ctr > 5), r_pos_sum), 0)
        r = np.clip(r_pos_sum - r_neg_sum, -3, 3)

        self.rew_queue.append([r])
        self.rew_queue.pop(0)

        if abs(r_pos_sum) > 3 or abs(r_neg_sum) > 3:
            print("!!WARNING!! REWARD IS ABOVE |3|, at step: {}  rpos = {}, rneg = {}".format(self.step_ctr, r_pos, r_neg))

        # Calculate relative positions of targets
        relative_target = self.target[0] - torso_pos[0], self.target[1] - torso_pos[1]

        # Assemble agent observation
        compiled_obs = scaled_joint_angles, torso_quat, torso_vel, relative_target, phases, contacts, list(self.joints_rads_low[:3]), list(self.joints_rads_high[:3])
        compiled_obs_flat = [item for sublist in compiled_obs for item in sublist]
        self.obs_queue.append(compiled_obs_flat)
        self.obs_queue.pop(0)

        aux_obs = []
        if self.config["obs_input"]:
            [aux_obs.extend(c) for c in self.obs_queue]
        if self.config["act_input"]:
            [aux_obs.extend(c) for c in self.act_queue]
        if self.config["rew_input"] > 0:
            [aux_obs.extend(c) for c in self.rew_queue]
        if self.config["latent_input"]:
            aux_obs.extend(self.randomized_params_list_norm)
        if self.config["step_counter"]:
            aux_obs.extend([(float(self.step_ctr) / self.config["max_steps"]) * 2 - 1])

        env_obs = np.array(aux_obs).astype(np.float32)

        done = self.step_ctr > self.config["max_steps"] or reached_target

        if np.abs(roll) > 1.57 or np.abs(pitch) > 1.57:
            print("WARNING!! Absolute roll and pitch values exceed bounds: roll: {}, pitch: {}".format(roll, pitch))
            done = True

        if abs(torso_pos[0]) > 6 or abs(torso_pos[1]) > 6 or abs(torso_pos[2]) > 2.5:
            print("WARNING: TORSO OUT OF RANGE!!")
            done = True

        return env_obs, r, done, {}

    def reset(self, force_randomize=None):
        if self.config["randomize_env"]:
            self.robot = self.load_robot()

        self.time_vector = np.random.rand(18)

        # Reset episodal vars
        self.step_ctr = 0
        self.episode_ctr += 1
        self.prev_yaw_dev = 0

        #self.config["target_spawn_mu"][0] = np.maximum(0., self.config["target_spawn_mu"][0] - 0.00005)
        #self.config["target_spawn_sigma"][0] = np.minimum(4., self.config["target_spawn_sigma"][0] + 0.00005)

        if self.config["velocity_control"]:
            self.target_vel_nn_input = np.random.rand() * 2 - 1
            self.config["target_vel"] = 0.5 * (self.target_vel_nn_input + 1) * (max(self.config["target_vel_range"]) - min(self.config["target_vel_range"])) + min(self.config["target_vel_range"])

        if self.config["force_target_velocity"]:
            self.config["target_vel"] = self.config["forced_target_velocity"]
            self.target_vel_nn_input = 2 * ((self.config["target_vel"] - min(self.config["target_vel_range"])) / (max(self.config["target_vel_range"]) - min(self.config["target_vel_range"]))) - 1

        # Change heightmap with small probability
        if np.random.rand() < self.config["env_change_prob"] and not self.config["terrain_name"] == "flat":
            self.terrain = self.generate_rnd_env()

        # Get heightmap height at robot position
        if self.terrain is None or self.config["terrain_name"] == "flat":
            spawn_height = 0
        else:
            spawn_height = 0.5 * np.max(self.terrain_hm[self.config["env_length"] // 2 - 3:self.config["env_length"] // 2 + 3, self.config["env_width"] // 2 - 3 : self.config["env_width"] // 2 + 3]) * self.config["mesh_scale_vert"]

        # Random initial rotation
        rnd_rot = np.random.rand() * 0.3 - 0.15
        rnd_quat = p.getQuaternionFromAxisAngle([0, 0, 1], rnd_rot)

        joint_init_pos_list = self.norm_to_rads([0] * 18)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(18)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, spawn_height + 0.15], rnd_quat, physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(18),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * 18,
                                    forces=[self.config["max_joint_force"]] * 18,
                                    physicsClientId=self.client_ID)

        self.update_targets()
        self.prev_target_dist = np.sqrt((0 - self.target[0]) ** 2 + (0 - self.target[1]) ** 2)
        tar_angle = np.arctan2(self.target[1] - 0, self.target[0] - 0)
        self.prev_yaw_deviation = np.min((abs((rnd_rot % 6.283) - (tar_angle % 6.283)), abs(rnd_rot - tar_angle)))

        for i in range(10):
            p.stepSimulation(physicsClientId=self.client_ID)

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs

    def test_agent(self, policy):
        import src.my_utils as my_utils
        for _ in range(100):
            obs = self.reset()
            cum_rew = 0
            ctr = 0
            while True:
                torso_pos_prev, torso_quat_prev, _, _, joint_angles_prev, _, _, _, _, _ = self.get_obs()
                action, _ = policy.sample_action(my_utils.to_tensor(obs, True))
                obs, reward, done, info = self.step(action.detach().squeeze(0).numpy())
                cum_rew += reward
                self.render()

                if ctr % 10 == 0 and ctr > 0 and True:
                    p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                                jointIndices=range(18),
                                                controlMode=p.POSITION_CONTROL,
                                                targetPositions=[0] * 18,
                                                forces=[0] * 18,
                                                physicsClientId=self.client_ID)
                    joint_angles_desired = self.norm_to_rads(np.tanh(action.detach().squeeze(0).numpy() * 0.5))
                    for _ in range(3):
                        [p.resetJointState(self.robot, k, joint_angles_prev[k], 0, physicsClientId=self.client_ID) for k in range(18)]
                        p.stepSimulation(physicsClientId=self.client_ID)
                        time.sleep(0.6)

                        [p.resetJointState(self.robot, k, joint_angles_desired[k], 0, physicsClientId=self.client_ID) for k in range(18)]
                        p.stepSimulation(physicsClientId=self.client_ID)
                        time.sleep(0.6)

                    [p.resetJointState(self.robot, k, joint_angles_prev[k], 0, physicsClientId=self.client_ID) for k in
                     range(18)]
                    p.stepSimulation(physicsClientId=self.client_ID)

                ctr += 1

                if done:
                    print(cum_rew)
                    break
        env.close()

    def test_leg_coordination(self):
        np.set_printoptions(precision=3)
        self.reset()
        n_steps = 30
        VERBOSE=True
        while True:
            t1 = time.time()
            sc = 1.0
            test_acts = [[0, 0, 0], [0, sc, sc], [0, -sc, -sc], [0, sc, -sc], [0, -sc, sc], [sc, 0, 0], [-sc, 0, 0]]
            for i, a in enumerate(test_acts):
                for j in range(n_steps):
                    #a = list(np.random.randn(3))
                    scaled_obs, _, _, _ = self.step(a * 12)
                _, _, _, _, joint_angles, _, joint_torques, contacts, ctct_torso = self.get_obs()
                if VERBOSE:
                    print("Obs rads: ", joint_angles)
                    print("Obs normed: ", self.rads_to_norm(joint_angles))
                    print("For action rads: ", self.norm_to_rads(a * 8))
                    print("action normed: ", a)
                    #input()

                #self.reset()

            t2 = time.time()
            print("Time taken for iteration: {}".format(t2 - t1))

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

if __name__ == "__main__":
    import yaml
    with open("configs/wp_flat.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = HexapodBulletEnv(env_config)
    env.test_leg_coordination()
