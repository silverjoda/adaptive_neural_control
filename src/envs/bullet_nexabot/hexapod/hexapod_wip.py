import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import random
import torch as T
import gym
from gym import spaces
from opensimplex import OpenSimplex


import torch as T
import torch.nn as nn
import torch.nn.functional as F

class ContactRegNN(nn.Module):
    def __init__(self, n_inputs=3, n_actions=2):
        nn.Module.__init__(self)
        n_hidden = 12
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)
        self.activ_fn = nn.Tanh()

    def forward(self, x):
        x = self.activ_fn(self.fc1(x))
        x = self.activ_fn(self.fc2(x))
        x = self.fc3(x)
        return x

class ContactRegManual(nn.Module):
    def __init__(self, n_inputs=3, n_actions=2):
        nn.Module.__init__(self)
        self.p1 = -0.5#nn.Parameter(T.tensor(-0.5), requires_grad=True)
        self.p2 = 0.3#nn.Parameter(T.tensor(0.3), requires_grad=True)

    def forward(self, x):
        c1 = T.le(x[:, 1], self.p1)
        c2 = T.le(x[:, 2], self.p2)
        out = T.cat((1. * T.logical_or(c1, c2).unsqueeze(1), 1. * (T.logical_and(T.logical_not(c1), T.logical_not(c2))).unsqueeze(1)), dim=1)
        return out

# To mirror quaternion along x-z plane (or y axis) just use q_mirror = [qx, -qy, qz, -qw]

class HexapodBulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, animate=False, max_steps=100, seed=None, step_counter=False, terrain_name=None, training_mode="straight", variable_velocity=False):
        if seed is not None:
            np.random.seed(seed)
            T.manual_seed(seed)

        if (animate):
            self.client_ID = p.connect(p.GUI)
            print(" --Starting GUI mode-- ")
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        self.animate = animate
        self.max_steps = max_steps
        self.seed = seed
        self.step_counter = step_counter
        self.terrain_name = terrain_name
        self.training_mode = training_mode
        self.variable_velocity = variable_velocity
        self.force_target_velocity = False
        self.replace_envs = False
        self.env_change_prob = 0.1
        self.walls = False

        # Simulation parameters
        self.env_width = 60
        self.env_length = self.max_steps
        self.max_joint_force = 1.3
        self.forced_target_vel = 0.2
        self.target_vel = 0.2
        self.target_vel_nn_input = 0
        self.target_vel_range = [0.1, 0.3]
        self.sim_steps_per_iter = 24
        self.mesh_scale_lat = 0.1 # 0.1
        self.mesh_scale_vert = 2
        self.lateral_friction = 1.2
        self.training_difficulty = 0.15 # 0.15
        self.training_difficulty_increment = 0.0001

        if self.terrain_name.startswith("stairs"):
            self.env_width *= 4
            self.env_length *= 4
            self.mesh_scale_lat /= 4
            self.target_vel = 0.15

        # Environment parameters
        self.obs_dim = 18 + 6 + 4 + int(step_counter) + int(variable_velocity)
        self.act_dim = 18
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        # Normal
        self.joints_rads_low = np.array([-0.4, -1.6, 0.9] * 6)
        self.joints_rads_high = np.array([0.4, -0.6, 1.9] * 6)

        if self.training_mode.endswith("extreme"):
            # Extreme
            self.joints_rads_low = np.array([-0.4, 0, -0.5] * 6)
            self.joints_rads_high = np.array([0.4, 1.0, 0.5] * 6)

        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.coxa_joint_ids = range(0, 18, 3)
        self.femur_joint_ids = range(1, 18, 3)
        self.tibia_joint_ids = range(2, 18, 3)
        self.left_joints_ids = [0,1,2,6,7,8,12,13,14]
        self.right_joints_ids = [3,4,5,9,10,11,15,16,17]

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.urdf_name = "hexapod_wip_normal.urdf"
        if self.training_mode.endswith("extreme"):
            self.urdf_name = "hexapod_wip_extreme.urdf"
        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.urdf_name), physicsClientId=self.client_ID)
        self.generate_rnd_env()

        # Change joint limits dynamically:
        # for i in range(18):
        #     p.changeDynamics(bodyUniqueId=self.robot, linkIndex=i, jointLowerLimit=self.joints_rads_low[i],
        #                      jointUpperLimit=self.joints_rads_high[i], physicsClientId=self.client_ID)

        # Change contact friction for legs and torso
        for i in range(6):
            p.changeDynamics(self.robot, 3 * i + 2, lateralFriction=self.lateral_friction)
        p.changeDynamics(self.robot, -1, lateralFriction=self.lateral_friction)

        # Episodal parameters
        self.step_ctr = 0
        self.episode_ctr = 0
        self.xd_queue = []
        self.joint_work_done_arr_list = []
        self.joint_angle_arr_list = []
        self.prev_yaw_dev = 0
        self.max_dist_travelled = 0

        # Check joint limits
        # read_joint_limits = [p.getJointInfo(self.robot, i, self.client_ID)[8:10] for i in range(18)]
        # for i in range(6):
        #     triplet = read_joint_limits[i*3:i*3+3]
        #     assert triplet[0] == (self.joints_rads_low[0],self.joints_rads_high[0])
        #     assert triplet[1] == (self.joints_rads_low[1], self.joints_rads_high[1])
        #     assert triplet[2] == (self.joints_rads_low[2], self.joints_rads_high[2])
        # print("Read joint limits: {}".format(read_joint_limits))

        # Contact regression
        # self.contact_reg_nn = ContactRegNN()
        # self.contact_reg_manual = ContactRegManual()
        # self.contact_reg_optim = T.optim.Adam(params=self.contact_reg_nn.parameters(), lr=0.001, weight_decay=0.0001)
        # #self.contact_reg_manual_optim = T.optim.Adam(params=self.contact_reg_manual.parameters(), lr=0.002)
        # self.contact_reg_manual_best_score = 1
        # self.contact_reg_manual_best_params = None
        # self.contact_reg_dataset = []
        # self.contact_reg_batchsize = 50

    def make_heightfield(self, height_map=None):
        if hasattr(self, 'terrain'):
            p.removeBody(self.terrain, self.client_ID)
        if height_map is None:
            heightfieldData = np.zeros(self.env_width * self.env_length)
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[self.mesh_scale_lat , self.mesh_scale_lat , self.mesh_scale_vert],
                                                  heightfieldTextureScaling=(self.env_width - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=self.env_length,
                                                  numHeightfieldColumns=self.env_width)
        else:
            heightfieldData = height_map.ravel(order='F')
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[self.mesh_scale_lat , self.mesh_scale_lat , self.mesh_scale_vert],
                                                  heightfieldTextureScaling=(self.env_width - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=height_map.shape[0],
                                                  numHeightfieldColumns=height_map.shape[1],
                                                  physicsClientId=self.client_ID)
        terrain = p.createMultiBody(0, terrainShape, physicsClientId=self.client_ID)
        p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_ID)
        return terrain

    def generate_heightmap(self, env_name):
        current_height = 0
        if env_name == "flat" or env_name is None:
            hm = np.ones((self.env_length, self.env_width)) * current_height

        if env_name == "tiles":
            sf = 3
            hm = np.random.randint(0, 20 * self.training_difficulty, # 15
                                   size=(self.env_length // sf, self.env_width // sf)).repeat(sf, axis=0).repeat(sf, axis=1)
            hm_pad = np.zeros((self.env_length, self.env_width))
            hm_pad[:hm.shape[0], :hm.shape[1]] = hm
            hm = hm_pad + current_height

        if env_name == "pipe":
            pipe_form = np.square(np.linspace(-1.2, 1.2, self.env_width))
            pipe_form = np.clip(pipe_form, 0, 1)
            hm = 255 * np.ones((self.env_width, self.env_length)) * pipe_form[np.newaxis, :].T
            hm += current_height

        if env_name == "stairs_up":
            hm = np.ones((self.env_length, self.env_width)) * current_height
            stair_height = 14 * self.training_difficulty
            stair_width = 8

            initial_offset = self.env_length // 2 - self.env_length // 14
            n_steps = min(math.floor(self.env_length / stair_width) - 1, 10)

            for i in range(n_steps):
                hm[initial_offset + i * stair_width: initial_offset + i * stair_width + stair_width, :] = current_height
                current_height += stair_height

            hm[n_steps * stair_width + initial_offset:, :] = current_height

        if env_name == "ramp_up":
            max_height = 200
            hm = np.ones((self.env_length, self.env_width)) * current_height
            initial_offset = self.env_length // 2 - self.env_length // 8
            hm[initial_offset:initial_offset + self.env_length // 4, :] = np.tile(np.linspace(current_height, current_height + max_height, self.env_length // 4)[:, np.newaxis], (1, self.env_width))
            current_height += max_height
            hm[initial_offset + self.env_length // 4:, :] = current_height

        if env_name == "stairs_down":
            stair_height = 14 * self.training_difficulty
            stair_width = 8

            initial_offset = self.env_length // 2
            n_steps = min(math.floor(self.env_length / stair_width) - 1, 10)

            current_height = n_steps * stair_height

            hm = np.ones((self.env_length, self.env_width)) * current_height

            for i in range(n_steps):
                hm[initial_offset + i * stair_width: initial_offset + i * stair_width + stair_width, :] = current_height
                current_height -= stair_height

            hm[n_steps * stair_width + initial_offset:, :] = current_height

        if env_name == "verts":
            wdiv = 4
            ldiv = 14
            hm = np.random.randint(0, 75,
                                   size=(self.env_width // wdiv, self.env_length // ldiv),
                                   dtype=np.uint8).repeat(wdiv, axis=0).repeat(ldiv, axis=1)
            hm[:, :50] = 0
            hm[hm < 50] = 0
            hm = 75 - hm

        if env_name == "triangles":
            cw = 10
            # Make even dimensions
            M = math.ceil(self.env_width)
            N = math.ceil(self.env_length)
            hm = np.zeros((M, N), dtype=np.float32)
            M_2 = math.ceil(M / 2)

            # Amount of 'tiles'
            Mt = 2
            Nt = int(self.env_length / 10.)
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

            height = 40 * self.training_difficulty # 30-40

            M = math.ceil(self.env_width)
            N = math.ceil(self.env_length)
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

    def generate_rnd_env(self):
        if self.terrain_name is None:
            self.terrain_hm = np.zeros((self.env_length, self.env_width))
            self.terrain = self.make_heightfield(self.terrain_hm)
            return 1, None, None

        self.terrain_hm, _ = self.generate_heightmap(self.terrain_name)
        self.terrain_hm /= 255.
        self.terrain = self.make_heightfield(self.terrain_hm)

    def get_obs(self):
        # Torso
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID) # xyz and quat: x,y,z,w
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)

        ctct_leg_1 = int(len(p.getContactPoints(self.robot, self.terrain, 2, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        ctct_leg_2 = int(len(p.getContactPoints(self.robot, self.terrain, 5, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        ctct_leg_3 = int(len(p.getContactPoints(self.robot, self.terrain, 8, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        ctct_leg_4 = int(len(p.getContactPoints(self.robot, self.terrain, 11, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        ctct_leg_5 = int(len(p.getContactPoints(self.robot, self.terrain, 14, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        ctct_leg_6 = int(len(p.getContactPoints(self.robot, self.terrain, 17, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        ctct_torso = int(len(p.getContactPoints(self.robot, self.terrain, -1, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1

        #contacts = [ctct_leg_1, ctct_leg_2, ctct_leg_3, ctct_leg_4, ctct_leg_5, ctct_leg_6]
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
        return torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso

    def scale_joints(self, joints):
        sjoints = np.array(joints)
        sjoints = ((sjoints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        return sjoints

    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    def render(self, close=False):
        pass

    def step(self, ctrl, render=False):
        ctrl_clipped = np.clip(ctrl, -1, 1)
        scaled_action = self.scale_action(ctrl_clipped)
        # p.setJointMotorControlArray(bodyUniqueId=self.robot,
        #                             jointIndices=range(18),
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPositions=scaled_action,
        #                             forces=[self.max_joint_force] * 18,
        #                             positionGains=[0.007] * 18,
        #                             velocityGains=[0.1] * 18,
        #                             physicsClientId=self.client_ID)

        for i in range(18):
            p.setJointMotorControl2(bodyUniqueId=self.robot,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=scaled_action[i],
                                    force=self.max_joint_force,
                                    positionGain=0.1,
                                    velocityGain=0.1,
                                    maxVelocity=2.0,
                                    physicsClientId=self.client_ID)

        leg_ctr = 0
        obs_sequential = []
        for i in range(self.sim_steps_per_iter):
            if leg_ctr < 6:
                obs_sequential.extend(p.getJointStates(self.robot, range(leg_ctr * 3, (leg_ctr + 1) * 3), physicsClientId=self.client_ID))
                leg_ctr += 1
            p.stepSimulation(physicsClientId=self.client_ID)
            if (self.animate or render) and True: time.sleep(0.0038)

        joint_angles_skewed = []
        for o in obs_sequential:
            joint_angles_skewed.append(o[0])

        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso = self.get_obs()
        xd, yd, zd = torso_vel
        thd, phid, psid = torso_angular_vel
        qx, qy, qz, qw = torso_quat

        scaled_joint_angles = self.scale_joints(joint_angles_skewed)
        scaled_joint_angles_true = self.scale_joints(joint_angles)
        scaled_joint_angles = np.clip(scaled_joint_angles, -2, 2)
        scaled_joint_angles_true = np.clip(scaled_joint_angles_true, -2, 2)
        #print(np.min(scaled_joint_angles), np.max(scaled_joint_angles))

        # self.joint_angle_arr_list.append(joint_angles)
        # joint_angle_arr_recent = np.array(self.joint_angle_arr_list[-15 - np.random.randint(0,20):])
        # joint_angle_arr_quantiles = np.quantile(joint_angle_arr_recent, [0.25,0.5,0.75], axis=0)
        # left_quantiles = joint_angle_arr_quantiles[:, self.left_joints_ids]
        # right_quantiles = joint_angle_arr_quantiles[:, self.right_joints_ids]
        # quantile_pen = np.mean(np.square(left_quantiles - right_quantiles))

        # Calculate work done by each motor
        joint_work_done_arr = np.array(joint_torques) * np.array(joint_velocities)
        self.joint_work_done_arr_list.append(joint_work_done_arr)

        # Instantaneous work penalties
        coxa_work_pen = np.mean(np.square(joint_work_done_arr[self.coxa_joint_ids]))
        femur_work_pen = np.mean(np.square(joint_work_done_arr[self.femur_joint_ids]))
        tibia_work_pen = np.mean(np.square(joint_work_done_arr[self.tibia_joint_ids]))
        total_work_pen = np.mean(np.square(joint_work_done_arr))

        # Cumulative mean work done per joint
        # joint_work_done_arr_recent = np.array(self.joint_work_done_arr_list[-15 - np.random.randint(0,20):])
        # joint_work_done_floating_avg = np.mean(joint_work_done_arr_recent, axis=0)
        #
        # # Symmetry penalty
        # left_work_mean = joint_work_done_floating_avg[self.left_joints_ids]
        # right_work_mean = joint_work_done_floating_avg[self.right_joints_ids]
        # symmetry_work_pen = np.mean(np.square(left_work_mean - right_work_mean))

        # Unsuitable position penalty
        unsuitable_position_pen = 0
        leg_pen = []
        for i in range(6):
            _, a1, a2 = scaled_joint_angles_true[i * 3: i * 3 + 3]
            pen = np.maximum((np.sign(a1) * (a1 ** 2)) * (np.sign(a2) * (a2 ** 2)), 0)
            unsuitable_position_pen += pen
            leg_pen.append(pen)

        # Contact penalty
        #contact_rew = np.mean((np.array(contacts) + 1. / 2.))

        # Torso contact penalty
        torso_contact_pen = (ctct_torso * 0.5 + 0.5)

        # Calculate yaw
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)
        q_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Velocity reward
        self.xd_queue.append(xd)
        if len(self.xd_queue) > 7:
            self.xd_queue.pop(0)
        xd_av = sum(self.xd_queue) / len(self.xd_queue)

        velocity_rew = 1. / (abs(xd_av - self.target_vel) + 1.) - 1. / (self.target_vel + 1.)
        velocity_rew *= (0.3 / self.target_vel)
        velocity_rew = velocity_rew / (1 + abs(q_yaw) * 15.) # scale velocity reward by yaw deviation

        yaw_improvement_reward = abs(self.prev_yaw_dev) - abs(q_yaw)
        self.prev_yaw_dev = q_yaw

        # Tmp spoofs
        quantile_pen = symmetry_work_pen = contact_rew = 0

        if self.training_mode == "straight":
            r_neg = {"pitch" : np.square(pitch) * 1.5 * self.training_difficulty,
                    "roll" : np.square(roll) * 0.5 * self.training_difficulty,
                    "zd" : np.square(zd) * 0.1 * self.training_difficulty,
                    "yd" : np.square(yd) * 0.5 * self.training_difficulty,
                    "phid": np.square(phid) * 0.02 * self.training_difficulty,
                    "thd": np.square(thd) * 0.02 * self.training_difficulty,
                    "quantile_pen" : quantile_pen * 0.0 * self.training_difficulty * (self.step_ctr > 10),
                    "symmetry_work_pen" : symmetry_work_pen * 0.0 * self.training_difficulty * (self.step_ctr > 10),
                    "torso_contact_pen": torso_contact_pen * 0.1 * self.training_difficulty,
                    "total_work_pen" : np.minimum(total_work_pen * 0.01 * self.training_difficulty * (self.step_ctr > 10), 1),
                    "unsuitable_position_pen" : unsuitable_position_pen * 0.00 * self.training_difficulty}
            r_pos = {"velocity_rew" : np.clip(velocity_rew * 4, -1, 1),
                     "yaw_improvement_reward" :  np.clip(yaw_improvement_reward * 3., -1, 1),
                     "contact_rew" : contact_rew * 0}
            r_pos_sum = sum(r_pos.values())
            r_neg_sum = sum(r_neg.values()) * (self.step_ctr > 5)
            r = np.clip(r_pos_sum - r_neg_sum, -3, 3)
            if abs(r_pos_sum) > 3 or abs(r_neg_sum) > 3:
                print("!!WARNING!! REWARD IS ABOVE |3|, at step: {}  rpos = {}, rneg = {}".format(self.step_ctr, r_pos, r_neg))
        elif self.training_mode.startswith("straight_rough"):
            r_neg = {"pitch": np.square(pitch) * 0.0 * self.training_difficulty,
                     "roll": np.square(roll) * 0.0 * self.training_difficulty,
                     "zd": np.square(zd) * 0.1 * self.training_difficulty,
                     "yd": np.square(yd) * 0.1 * self.training_difficulty,
                     "phid": np.square(phid) * 0.02 * self.training_difficulty,
                     "thd": np.square(thd) * 0.02 * self.training_difficulty,
                     "quantile_pen": quantile_pen * 0.0 * self.training_difficulty * (self.step_ctr > 10),
                     "symmetry_work_pen": symmetry_work_pen * 0.00 * self.training_difficulty * (self.step_ctr > 10),
                     "torso_contact_pen" : torso_contact_pen * 0.0 * self.training_difficulty,
                     "total_work_pen": np.minimum(
                         total_work_pen * 0.03 * self.training_difficulty * (self.step_ctr > 10), 1),
                     "unsuitable_position_pen": unsuitable_position_pen * 0.0 * self.training_difficulty}
            r_pos = {"velocity_rew": np.clip(velocity_rew * 4, -1, 1),
                     "yaw_improvement_reward": np.clip(yaw_improvement_reward * 1.0, -1, 1)}
            r_pos_sum = sum(r_pos.values())
            r_neg_sum = sum(r_neg.values()) * (self.step_ctr > 10)
            r = np.clip(r_pos_sum - r_neg_sum, -3, 3)
            if abs(r_pos_sum) > 3 or abs(r_neg_sum) > 3:
                print("!!WARNING!! REWARD IS ABOVE |3|, at step: {}  rpos = {}, rneg = {}".format(self.step_ctr, r_pos, r_neg))
        elif self.training_mode == "turn_left":
            r_neg = torso_contact_pen * 0.2 + np.square(pitch) * 0.2 + np.square(roll) * 0.2 + unsuitable_position_pen * 0.1
            r_pos = torso_angular_vel[2] * 1.
            r = np.clip(r_pos - r_neg, -3, 3)
        elif self.training_mode == "turn_right":
            r_neg = torso_contact_pen * 0.2 + np.square(pitch) * 0.2 + np.square(
                roll) * 0.2 + unsuitable_position_pen * 0.1
            r_pos = -torso_angular_vel[2] * 1.
            r = np.clip(r_pos - r_neg, -3, 3)
        elif self.training_mode == "stairs":
            r_neg = {"pitch": np.square(pitch) * 0.0 * self.training_difficulty,
                     "roll": np.square(roll) * 0.0 * self.training_difficulty,
                     "zd": np.square(zd) * 0.0 * self.training_difficulty,
                     "yd": np.square(yd) * 0.0 * self.training_difficulty,
                     "phid": np.square(phid) * 0.00 * self.training_difficulty,
                     "thd": np.square(thd) * 0.0 * self.training_difficulty,
                     "quantile_pen": quantile_pen * 0.0 * self.training_difficulty * (self.step_ctr > 10),
                     "symmetry_work_pen": symmetry_work_pen * 0.00 * self.training_difficulty * (self.step_ctr > 10),
                     "torso_contact_pen": torso_contact_pen * 0.0 * self.training_difficulty,
                     "total_work_pen": np.minimum(
                         total_work_pen * 0.0 * self.training_difficulty * (self.step_ctr > 10), 1),
                     "unsuitable_position_pen": unsuitable_position_pen * 0.0 * self.training_difficulty}
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

        if self.step_counter:
            env_obs = np.concatenate((env_obs, [self.step_encoding]))

        if self.variable_velocity:
            env_obs = np.concatenate((env_obs, [self.target_vel_nn_input]))

        self.step_ctr += 1
        self.step_encoding = (float(self.step_ctr) / self.max_steps) * 2 - 1

        if torso_pos[0] > self.max_dist_travelled:
            self.max_dist_travelled += 0.05

        done = self.step_ctr > self.max_steps # or np.abs(roll) > 1.57 or np.abs(pitch) > 1.57

        if np.abs(roll) > 1.57 or np.abs(pitch) > 1.57:
            print("WARNING!! Absolute roll and pitch values exceed bounds: roll: {}, pitch: {}".format(roll, pitch))

        if abs(torso_pos[0]) > 3 or abs(torso_pos[1]) > 2.5 or abs(torso_pos[2]) > 2.5:
            print("WARNING: TORSO OUT OF RANGE!!")
            done = True

        return env_obs, r, done, {}

    def reset(self):
        # Reset episodal vars
        self.step_ctr = 0
        self.episode_ctr += 1
        self.xd_queue = []
        self.joint_work_done_arr_list = []
        self.joint_angle_arr_list = []
        self.prev_yaw_dev = 0
        #self.training_difficulty = np.clip(self.max_dist_travelled - 1.0, 0, 1)
        self.training_difficulty = np.minimum(self.training_difficulty + self.training_difficulty_increment, 1.0)
        self.max_dist_travelled = self.max_dist_travelled * 0.95

        # Change contact friction for legs and torso
        # self.lateral_friction = np.random.rand(0.7) + 0.7
        # for i in range(6):
        #     p.changeDynamics(self.robot, 3 * i + 2, lateralFriction=self.lateral_friction)
        # p.changeDynamics(self.robot, -1, lateralFriction=self.lateral_friction)

        # Calculate target velocity
        if self.variable_velocity:
            self.target_vel_nn_input = np.random.rand() * 2 - 1
            self.target_vel = 0.5 * (self.target_vel_nn_input + 1) * (max(self.target_vel_range) - min(self.target_vel_range)) + min(self.target_vel_range)

        if self.force_target_velocity or True:
            self.target_vel = self.forced_target_vel
            self.target_vel_nn_input = 2 * ((self.target_vel - min(self.target_vel_range)) / (max(self.target_vel_range) - min(self.target_vel_range))) - 1

        if self.episode_ctr % 100 == 0 and self.episode_ctr > 100:
            print("--------- CURRENT TRAINING DIFFICULTY: {}".format(self.training_difficulty))

        # Calculate encoding for current step
        self.step_encoding = (float(self.step_ctr) / self.max_steps) * 2 - 1

        # Change heightmap with small probability
        if np.random.rand() < self.env_change_prob:
            self.generate_rnd_env()

        # Get heightmap height at robot position
        if self.terrain is None:
            spawn_height = 0
        else:
            spawn_height = 0.5 * np.max(self.terrain_hm[self.env_length // 2 - 3:self.env_length // 2 + 3, self.env_width // 2 - 3 : self.env_width // 2 + 3]) * self.mesh_scale_vert

        # Random initial rotation
        rnd_rot = np.random.rand() * 1.0 - 0.5
        rnd_quat = p.getQuaternionFromAxisAngle([0,0,1], rnd_rot)
        self.prev_yaw_dev = rnd_rot

        joint_init_pos_list = self.scale_action([0] * 18)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(18)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, spawn_height + 0.15], rnd_quat, physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(18),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * 18,
                                    forces=[self.max_joint_force] * 18,
                                    physicsClientId=self.client_ID)

        for i in range(30):
            p.stepSimulation(physicsClientId=self.client_ID)

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs

    def test(self, policy, seed=None):
        if seed is not None:
            np.random.seed(seed)
        total_rew = 0
        for i in range(100):
            obs = self.reset()
            cr = 0
            for j in range(self.max_steps):
                nn_obs = T.FloatTensor(obs).unsqueeze(0)
                action = policy(nn_obs).detach()
                obs, r, done, od, = self.step(action[0].numpy(), render=True)
                cr += r
                total_rew += r
            print("Total episode reward: {}".format(cr))
        print("Total reward: {}".format(total_rew))

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
                _, _, _, _, joint_angles, _, joint_torques, contacts, ctct_torso = self.get_obs()
                if VERBOSE:
                    print("Obs rads: ", joint_angles)
                    print("Obs normed: ", self.scale_joints(joint_angles))
                    print("For action rads: ", self.scale_action(a * 6))
                    print("action normed: ", a)
                    #input()

            t2 = time.time()
            print("Time taken for iteration: {}".format(t2 - t1))

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

if __name__ == "__main__":
    env = HexapodBulletEnv(animate=True, terrain_name="stairs_up")
    env.test_leg_coordination()
