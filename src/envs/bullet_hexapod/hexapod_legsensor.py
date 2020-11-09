import math
import os
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
import torch as T
import torch.nn as nn
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
            np.random.seed(self.seed)
            T.manual_seed(self.seed)
        else:
            rnd_seed = int((time.time() % 1) * 10000000)
            np.random.seed(rnd_seed)
            T.manual_seed(rnd_seed + 1)

        if (self.config["animate"]):
            self.client_ID = p.connect(p.GUI)
            print(" --Starting GUI mode-- ")
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        if self.config["terrain_name"].startswith("stairs"):
            self.config["env_width"] *= 4
            self.config["max_steps"] *= 4
            self.config["mesh_scale_lat"] /= 4
            self.config["target_vel"] = 0.15

        # Environment parameters
        self.obs_dim = 18 + 6 + 4 + int(self.config["step_counter"]) + int(self.config["velocity_control"])
        self.act_dim = 18
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        # Normal mode
        self.joints_rads_low = np.array(config["joints_rads_low"] * 6)
        self.joints_rads_high = np.array(config["joints_rads_high"] * 6)

        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.coxa_joint_ids = range(0, 18, 3)
        self.femur_joint_ids = range(1, 18, 3)
        self.tibia_joint_ids = range(2, 18, 3)
        self.left_joints_ids = [0,1,2,8,9,10,15,16,17]
        self.right_joints_ids = [4,5,6,12,13,14,19,20,21]
        self.total_joint_ids = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.urdf_name = config["urdf_name"]

        self.robot = None
        self.robot = self.load_robot()

        if config["terrain_name"] == "flat":
            self.terrain = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)
        else:
            self.terrain = self.generate_rnd_env()

        # Change contact friction for legs and torso
        for i in range(6):
            p.changeDynamics(self.robot, 4 * i + 3, lateralFriction=self.config["lateral_friction"], physicsClientId=self.client_ID)
        p.changeDynamics(self.robot, -1, lateralFriction=self.config["lateral_friction"], physicsClientId=self.client_ID)

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

        #contacts = [int(len(p.getContactPoints(self.robot, self.terrain, i * 4 + 3, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1 for i in range(6)]
        ctct_torso = int(len(p.getContactPoints(self.robot, self.terrain, -1, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1

        contacts = np.zeros(6)

        # Joints
        obs = p.getJointStates(self.robot, range(18), physicsClientId=self.client_ID) # pos, vel, reaction(6), prev_torque
        joint_angles = []
        joint_velocities = []
        joint_torques = []
        for o in obs:
            joint_angles.append(o[0])
            joint_velocities.append(o[1])
            joint_torques.append(o[3])

        # Legtip velocities
        tip_link_ids = [3, 7, 11, 15, 19, 23]  #
        link_states = p.getLinkStates(bodyUniqueId=self.robot, linkIndices=tip_link_ids, computeLinkVelocity=True,
                                      physicsClientId=self.client_ID)
        tip_velocities = [o[6] for o in link_states]

        return torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso, tip_velocities

    def rads_to_norm(self, joints):
        sjoints = np.array(joints)
        sjoints = ((sjoints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        return sjoints

    def norm_to_rads(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    def render(self, close=False):
        pass

    def load_robot(self):
        # Remove old robot
        if self.robot is not None:
            p.removeBody(self.robot, physicsClientId=self.client_ID)

        # Randomize robot params
        self.randomized_robot_params = {"mass": 1 + (np.random.rand() * 1.0 - 0.5) * self.config["randomize_env"],
                                        "max_actuator_velocity": self.config["max_actuator_velocity"] + (np.random.rand() * 2.0 - 1.0) * self.config["randomize_env"],
                                        "lateral_friction": self.config["lateral_friction"] + (np.random.rand() * 1.0 - 0.5) * self.config["randomize_env"],
                                        "max_joint_force": self.config["max_joint_force"] + np.random.rand() * 1. * self.config["randomize_env"]}

        robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.urdf_name), physicsClientId=self.client_ID)

        if self.config["randomize_env"]:
            # Change base mass
            p.changeDynamics(robot, -1, mass=self.randomized_robot_params["mass"], physicsClientId=self.client_ID)

        for i in range(6):
            p.changeDynamics(robot, 3 * i + 2, lateralFriction=self.randomized_robot_params["lateral_friction"], physicsClientId=self.client_ID)
        p.changeDynamics(robot, -1, lateralFriction=self.randomized_robot_params["lateral_friction"], physicsClientId=self.client_ID)

        return robot

    def step(self, ctrl, render=False):
        if np.max(ctrl) > 1:
            ctrl_clipped = ctrl / np.abs(np.max(ctrl))
        else:
            ctrl_clipped = ctrl

        #ctrl_clipped = np.clip(np.array(ctrl) * self.config["action_scaler"], -1, 1)
        ctrl_clipped = np.tanh(np.array(ctrl) * self.config["action_scaler"])
        scaled_action = self.norm_to_rads(ctrl_clipped)

        for i in range(18):
            p.setJointMotorControl2(bodyUniqueId=self.robot,
                                    jointIndex=self.total_joint_ids[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=scaled_action[i],
                                    force=self.randomized_robot_params["max_joint_force"],
                                    positionGain=0.3,
                                    velocityGain=0.3,
                                    maxVelocity=self.randomized_robot_params["max_actuator_velocity"],
                                    physicsClientId=self.client_ID)

        # Read out joint angles sequentially (to simulate servo daisy chain delay)
        leg_ctr = 0
        obs_sequential = []
        for i in range(self.config["sim_steps_per_iter"]):
            if leg_ctr < 6:
                obs_sequential.extend(p.getJointStates(self.robot, range(leg_ctr * 4, ((leg_ctr + 1) * 4) - 1), physicsClientId=self.client_ID))
                leg_ctr += 1
            p.stepSimulation(physicsClientId=self.client_ID)
            if (self.config["animate"] or render) and True: time.sleep(0.00417)

        joint_angles_skewed = []
        for o in obs_sequential:
            joint_angles_skewed.append(o[0])

        # Get all observations
        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso, tip_velocities = self.get_obs()
        xd, yd, zd = torso_vel
        thd, phid, psid = torso_angular_vel
        qx, qy, qz, qw = torso_quat

        scaled_joint_angles = self.rads_to_norm(joint_angles_skewed) # Change back to skewed here
        scaled_joint_angles_true = self.rads_to_norm(joint_angles)
        scaled_joint_angles = np.clip(scaled_joint_angles, -2, 2)
        scaled_joint_angles_true = np.clip(scaled_joint_angles_true, -2, 2)

        # Calculate work done by each motor
        joint_work_done_arr = np.array(joint_torques) * np.array(joint_velocities)
        total_work_pen = np.mean(np.square(joint_work_done_arr))

        # Unsuitable position penalty
        unsuitable_position_pen = 0
        # leg_pen = []
        # for i in range(6):
        #     _, a1, a2 = scaled_joint_angles_true[i * 3: i * 3 + 3]
        #     pen = np.maximum((np.sign(a1) * (a1 ** 2)) * (np.sign(a2) * (a2 ** 2)), 0)
        #     unsuitable_position_pen += pen
        #     leg_pen.append(pen)

        # Calculate yaw
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        # Velocity reward
        # self.xd_queue.append(xd)
        # if len(self.xd_queue) > 7:
        #     self.xd_queue.pop(0)
        # xd_av = sum(self.xd_queue) / len(self.xd_queue)

        velocity_rew = np.minimum(xd, self.config["target_vel"]) / self.config["target_vel"]
        yaw_improvement_reward = np.square(self.prev_yaw_dev - yaw)
        self.prev_yaw_dev = yaw

        # Tmp spoofs
        quantile_pen = symmetry_work_pen = torso_contact_pen = 0

        # Calculate shuffle penalty

        shuffle_pen = np.sum([(np.square(x_tip) + np.square(y_tip)) * (contacts[k] == 1) for k, (x_tip, y_tip, _) in enumerate(tip_velocities)])

        if self.config["training_mode"] == "straight":
            r_neg = {"pitch" : np.square(pitch) * 1.5,
                     "roll": np.square(roll) * 1.5,
                     "shuffle_pen" : shuffle_pen * 0.1,
                     "yaw_pen" : np.square(yaw) * 0.7}

            r_pos = {"velocity_rew": np.clip(velocity_rew * 1, -1, 1),
                     "yaw_improvement_reward" :  np.clip(yaw_improvement_reward * 0.1, -1, 1)}

            r_pos_sum = sum(r_pos.values())
            r_neg_sum = np.maximum(np.minimum(sum(r_neg.values()) * (self.step_ctr > 5) * 1, r_pos_sum), 0)
            r = np.clip(r_pos_sum - r_neg_sum, -3, 3)
            if abs(r_pos_sum) > 3 or abs(r_neg_sum) > 3:
                print("!!WARNING!! REWARD IS ABOVE |3|, at step: {}  rpos = {}, rneg = {}".format(self.step_ctr, r_pos, r_neg))
        elif self.config["training_mode"].startswith("straight_rough"):
            r_neg = {"pitch": np.square(pitch) * 0.0 * self.config["training_difficulty"],
                     "roll": np.square(roll) * 0.0 * self.config["training_difficulty"],
                     "zd": np.square(zd) * 0.03 * self.config["training_difficulty"], # 0.1
                     "yd": np.square(yd) * 0.00 * self.config["training_difficulty"], # 0.1
                     "phid": np.square(phid) * 0.01 * self.config["training_difficulty"], # 0.02
                     "thd": np.square(thd) * 0.01 * self.config["training_difficulty"], # 0.02
                     "quantile_pen": quantile_pen * 0.0 * self.config["training_difficulty"] * (self.step_ctr > 10),
                     "symmetry_work_pen": symmetry_work_pen * 0.00 * self.config["training_difficulty"] * (self.step_ctr > 10),
                     "torso_contact_pen" : torso_contact_pen * 0.0 * self.config["training_difficulty"],
                     "total_work_pen": np.minimum(
                         total_work_pen * 0.00 * self.config["training_difficulty"] * (self.step_ctr > 10), 1), # 0.02
                     "unsuitable_position_pen": unsuitable_position_pen * 0.0 * self.config["training_difficulty"],
                     "shuffle_pen" : shuffle_pen * 0.07}
            r_pos = {"velocity_rew": np.clip(velocity_rew * 1, -1, 1),
                     "yaw_improvement_reward" :  np.clip(yaw_improvement_reward * 0.1, -1, 1)}
            r_pos_sum = sum(r_pos.values())
            r_neg_sum = np.maximum(np.minimum(sum(r_neg.values()) * (self.step_ctr > 10) * 1, r_pos_sum), 0)
            r = np.clip(r_pos_sum - r_neg_sum, -3, 3)
            if abs(r_pos_sum) > 3 or abs(r_neg_sum) > 3:
                print("!!WARNING!! REWARD IS ABOVE |3|, at step: {}  rpos = {}, rneg = {}".format(self.step_ctr, r_pos, r_neg))
        elif self.config["training_mode"] == "straight_no_pen":
            r_neg = {"pitch": np.square(pitch) * 0.0 * self.config["training_difficulty"],
                     "roll": np.square(roll) * 0.0 * self.config["training_difficulty"],
                     "zd": np.square(zd) * 0.0 * self.config["training_difficulty"], # 0.1
                     "yd": np.square(yd) * 0.0 * self.config["training_difficulty"], # 0.1
                     "phid": np.square(phid) * 0.0 * self.config["training_difficulty"], # 0.02
                     "thd": np.square(thd) * 0.0 * self.config["training_difficulty"], # 0.02
                     "quantile_pen": quantile_pen * 0.0 * self.config["training_difficulty"] * (self.step_ctr > 10),
                     "symmetry_work_pen": symmetry_work_pen * 0.00 * self.config["training_difficulty"] * (self.step_ctr > 10),
                     "torso_contact_pen" : torso_contact_pen * 0.0 * self.config["training_difficulty"],
                     "total_work_pen": np.minimum(
                         total_work_pen * 0.00 * self.config["training_difficulty"] * (self.step_ctr > 10), 1), # 0.03
                     "unsuitable_position_pen": unsuitable_position_pen * 0.0 * self.config["training_difficulty"]}
            r_pos = {"velocity_rew": np.clip(velocity_rew * 4, -1, 1),
                     "yaw_improvement_reward": np.clip(yaw_improvement_reward * 1.0, -1, 1)}
            r_pos_sum = sum(r_pos.values())
            r_neg_sum = np.maximum(np.minimum(sum(r_neg.values()) * (self.step_ctr > 10) * 1, r_pos_sum), 0)
            r = np.clip(r_pos_sum - r_neg_sum, -3, 3)
            if abs(r_pos_sum) > 3 or abs(r_neg_sum) > 3:
                print("!!WARNING!! REWARD IS ABOVE |3|, at step: {}  rpos = {}, rneg = {}".format(self.step_ctr, r_pos, r_neg))
        elif self.config["training_mode"] == "turn_left":
            r_neg = torso_contact_pen * 0.2 + np.square(pitch) * 0.2 + np.square(roll) * 0.2 + unsuitable_position_pen * 0.1
            r_pos = torso_angular_vel[2] * 1.
            r = np.clip(r_pos - r_neg, -3, 3)
        elif self.config["training_mode"] == "turn_right":
            r_neg = torso_contact_pen * 0.2 + np.square(pitch) * 0.2 + np.square(
                roll) * 0.2 + unsuitable_position_pen * 0.1
            r_pos = -torso_angular_vel[2] * 1.
            r = np.clip(r_pos - r_neg, -3, 3)
        elif self.config["training_mode"].startswith("stairs"):
            r_neg = {"pitch": np.square(pitch) * 0.0 * self.config["training_difficulty"],
                     "roll": np.square(roll) * 0.0 * self.config["training_difficulty"],
                     "zd": np.square(zd) * 0.0 * self.config["training_difficulty"],
                     "yd": np.square(yd) * 0.0 * self.config["training_difficulty"],
                     "phid": np.square(phid) * 0.00 * self.config["training_difficulty"],
                     "thd": np.square(thd) * 0.0 * self.config["training_difficulty"],
                     "quantile_pen": quantile_pen * 0.0 * self.config["training_difficulty"] * (self.step_ctr > 10),
                     "symmetry_work_pen": symmetry_work_pen * 0.00 * self.config["training_difficulty"] * (self.step_ctr > 10),
                     "torso_contact_pen": torso_contact_pen * 0.0 * self.config["training_difficulty"],
                     "total_work_pen": np.minimum(
                         total_work_pen * 0.0 * self.config["training_difficulty"] * (self.step_ctr > 10), 1),
                     "unsuitable_position_pen": unsuitable_position_pen * 0.0 * self.config["training_difficulty"]}
            r_pos = {"velocity_rew": np.clip(velocity_rew * 6, -1, 1),
                     "yaw_improvement_reward": np.clip(yaw_improvement_reward * 0.5, -1, 1)}
            r_pos_sum = sum(r_pos.values())
            r_neg_sum = sum(r_neg.values()) * (self.step_ctr > 10)
            r = np.clip(r_pos_sum - r_neg_sum, -3, 3)
            if abs(r_pos_sum) > 3 or abs(r_neg_sum) > 3:
                print("!!WARNING!! REWARD IS ABOVE |3|, at step: {}  rpos = {}, rneg = {}".format(self.step_ctr, r_pos,
                                                                                                  r_neg))
        else:
            print("No mode selected")
            exit()

        # Assemble agent observation
        env_obs = np.concatenate((scaled_joint_angles, torso_quat, contacts))

        if self.config["step_counter"]:
            env_obs = np.concatenate((env_obs, [self.step_encoding]))

        if self.config["velocity_control"]:
            env_obs = np.concatenate((env_obs, [self.target_vel_nn_input]))

        self.step_ctr += 1
        self.step_encoding = (float(self.step_ctr) / self.config["max_steps"]) * 2 - 1

        done = self.step_ctr > self.config["max_steps"]

        if np.abs(roll) > 1.57 or np.abs(pitch) > 1.57:
            print("WARNING!! Absolute roll and pitch values exceed bounds: roll: {}, pitch: {}".format(roll, pitch))
            done = True

        if abs(torso_pos[0]) > 6 or abs(torso_pos[1]) > 3 or abs(torso_pos[2]) > 2.5:
            print("WARNING: TORSO OUT OF RANGE!!")
            done = True

        # for i in range(6):
        #     if scaled_joint_angles[i * 3 + 1] > .5 and scaled_joint_angles[i * 3 + 2] > .8:
        #         done = True
        #         r -= 1.
        #         break

        return env_obs.astype(np.float32), r, done, {}

    def reset(self, force_randomize=None):
        if self.config["randomize_env"]:
            self.robot = self.load_robot()

        # Reset episodal vars
        self.step_ctr = 0
        self.episode_ctr += 1
        self.xd_queue = []
        self.joint_work_done_arr_list = []
        self.joint_angle_arr_list = []
        self.prev_yaw_dev = 0

        if self.config["velocity_control"]:
            self.target_vel_nn_input = np.random.rand() * 2 - 1
            self.config["target_vel"] = 0.5 * (self.target_vel_nn_input + 1) * (max(self.config["target_vel_range"]) - min(self.config["target_vel_range"])) + min(self.config["target_vel_range"])

        if self.config["force_target_velocity"]:
            self.config["target_vel"] = self.config["forced_target_velocity"]
            self.target_vel_nn_input = 2 * ((self.config["target_vel"] - min(self.config["target_vel_range"])) / (max(self.config["target_vel_range"]) - min(self.config["target_vel_range"]))) - 1

        # Calculate encoding for current step
        self.step_encoding = (float(self.step_ctr) / self.config["max_steps"]) * 2 - 1

        # Change heightmap with small probability
        if np.random.rand() < self.config["env_change_prob"] and not self.config["terrain_name"] == "flat":
            self.terrain = self.generate_rnd_env()

        # Get heightmap height at robot position
        if self.terrain is None or self.config["terrain_name"] == "flat":
            spawn_height = 0
        else:
            spawn_height = 0.5 * np.max(self.terrain_hm[self.config["env_length"] // 2 - 3:self.config["env_length"] // 2 + 3, self.config["env_width"] // 2 - 3 : self.config["env_width"] // 2 + 3]) * self.config["mesh_scale_vert"]

        # Random initial rotation
        rnd_rot = np.random.rand() * 0.6 - 0.3
        rnd_quat = p.getQuaternionFromAxisAngle([0, 0, 1], rnd_rot)
        rnd_quat2 = p.getQuaternionFromEuler([0, 0, rnd_rot]) # JOOI, remove later
        assert np.isclose(rnd_quat, rnd_quat2, rtol=0.001).all(), print(rnd_quat, rnd_quat2) # JOOI, remove later
        self.prev_yaw_dev = rnd_rot

        joint_init_pos_list = self.norm_to_rads([0] * 18)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(18)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, spawn_height + 0.15], rnd_quat, physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(18),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * 18,
                                    forces=[self.config["max_joint_force"]] * 18,
                                    physicsClientId=self.client_ID)

        for i in range(30):
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
                    scaled_obs, _, _, _ = self.step(a * 6)
                _, _, _, _, joint_angles, _, joint_torques, contacts, ctct_torso, _ = self.get_obs()
                if VERBOSE:
                    print("Obs rads: ", joint_angles)
                    print("Obs normed: ", self.rads_to_norm(joint_angles))
                    print("For action rads: ", self.norm_to_rads(a * 6))
                    print("action normed: ", a)
                    #input()

                #self.reset()

            t2 = time.time()
            print("Time taken for iteration: {}".format(t2 - t1))

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = HexapodBulletEnv(env_config)
    env.test_leg_coordination()
