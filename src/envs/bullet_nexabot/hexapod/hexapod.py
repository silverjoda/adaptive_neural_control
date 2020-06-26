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

class HexapodBulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, animate=False, max_steps=200, seed=None, step_counter=False, env_list=None):
        if seed is not None:
            np.random.seed(seed)
            T.manual_seed(seed)

        if (animate):
            self.client_ID = p.connect(p.GUI)
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        self.animate = animate
        self.max_steps = max_steps
        self.seed = seed
        self.step_counter = step_counter
        self.env_list = env_list
        self.replace_envs = False
        self.n_envs = 1
        self.env_width = 40
        self.env_length = self.max_steps
        self.env_change_prob = 0.1
        self.walls = True

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hexapod.urdf"), physicsClientId=self.client_ID)
        self.generate_hybrid_env(self.n_envs, self.env_length)

        self.obs_dim = 28 + int(step_counter)
        self.act_dim = 18

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        self.joints_rads_low = np.array([-0.3, -1.6, 0.7] * 6)
        self.joints_rads_high = np.array([0.3, 0.0, 1.9] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.max_joint_force = 1.4
        self.target_vel = 0.2
        self.sim_steps_per_iter = 10
        self.step_ctr = 0
        self.xd_queue = []

    def make_heightfield(self, height_map=None):
        if hasattr(self, 'terrain'):
            p.removeBody(self.terrain, self.client_ID)

        if height_map is None:
            heightfieldData = np.zeros(self.env_width*self.env_length)
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.05, .05, 1],
                                                  heightfieldTextureScaling=(self.env_width - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=self.env_width,
                                                  numHeightfieldColumns=self.env_length)
        else:
            heightfieldData = height_map.ravel()
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.1, .1, 1],
                                                  heightfieldTextureScaling=(self.env_width - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=height_map.shape[0],
                                                  numHeightfieldColumns=height_map.shape[1])
        terrain = p.createMultiBody(0, terrainShape)
        p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_ID)

        return terrain

    def generate_hybrid_env(self, n_envs, steps):
        if self.env_list is None:
            self.terrain = self.make_heightfield(np.zeros((self.env_length, self.env_width)))
            return 1, None, None
        envs = np.random.choice(self.env_list, n_envs, replace=self.replace_envs)

        if n_envs == 1:
            size_list = [steps]
            scaled_indeces_list = [0]
        else:
            size_list = []
            raw_indeces = np.linspace(0, 1, n_envs + 1)[1:-1]
            current_idx = 0
            scaled_indeces_list = []
            for idx in raw_indeces:
                idx_scaled = int(steps * idx) + np.random.randint(0, int(steps / 6)) - int(steps / 12)
                scaled_indeces_list.append(idx_scaled)
                size_list.append(idx_scaled - current_idx)
                current_idx = idx_scaled
            size_list.append(steps - sum(size_list))

        maplist = []
        current_height = 0
        for m, s in zip(envs, size_list):
            hm, current_height = self.generate_heightmap(m, s, current_height)
            maplist.append(hm)
        total_hm = np.concatenate(maplist, 1)
        heighest_point = np.max(total_hm)
        height_SF = max(heighest_point / 255., 1)
        total_hm /= height_SF
        total_hm = np.clip(total_hm, 0, 255).astype(np.uint8)

        # Smoothen transitions
        bnd = 2
        if self.n_envs > 1:
            for s in scaled_indeces_list:
                total_hm_copy = np.array(total_hm)
                for i in range(s - bnd, s + bnd):
                    total_hm_copy[:, i] = np.mean(total_hm[:, i - bnd:i + bnd], axis=1)
                total_hm = total_hm_copy

        if self.walls:
            total_hm[0, :] = 255
            total_hm[:, 0] = 255
            total_hm[-1, :] = 255
            total_hm[:, -1] = 255
        else:
            total_hm[0, 0] = 255

        total_hm = total_hm.astype(np.float32) / 255.
        self.terrain = self.make_heightfield(total_hm)
        return envs, size_list, scaled_indeces_list

    def generate_heightmap(self, env_name, env_length, current_height):
        if env_name == "flat":
            hm = np.ones((self.env_width, env_length)) * current_height

        if env_name == "tiles":
            sf = 3
            hm = np.random.randint(0, 55,
                                   size=(self.env_width // sf, env_length // sf)).repeat(sf, axis=0).repeat(sf, axis=1)
            hm_pad = np.zeros((self.env_width, env_length))
            hm_pad[:hm.shape[0], :hm.shape[1]] = hm
            hm = hm_pad + current_height

        if env_name == "pipe":
            pipe_form = np.square(np.linspace(-1.2, 1.2, self.env_width))
            pipe_form = np.clip(pipe_form, 0, 1)
            hm = 255 * np.ones((self.env_width, env_length)) * pipe_form[np.newaxis, :].T
            hm += current_height

        if env_name == "stairs":
            hm = np.ones((self.env_width, env_length)) * current_height
            stair_height = 45
            stair_width = 4

            initial_offset = 0
            n_steps = math.floor(env_length / stair_width) - 1

            for i in range(n_steps):
                hm[:, initial_offset + i * stair_width: initial_offset + i * stair_width + stair_width] = current_height
                current_height += stair_height

            hm[:, n_steps * stair_width:] = current_height

        if env_name == "verts":
            wdiv = 4
            ldiv = 14
            hm = np.random.randint(0, 75,
                                   size=(self.env_width // wdiv, env_length // ldiv),
                                   dtype=np.uint8).repeat(wdiv, axis=0).repeat(ldiv, axis=1)
            hm[:, :50] = 0
            hm[hm < 50] = 0
            hm = 75 - hm

        if env_name == "triangles":
            cw = 10
            # Make even dimensions
            M = math.ceil(self.env_width)
            N = math.ceil(env_length)
            hm = np.zeros((M, N), dtype=np.float32)
            M_2 = math.ceil(M / 2)

            # Amount of 'tiles'
            Mt = 2
            Nt = int(env_length / 10.)
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

            height = 100

            M = math.ceil(self.env_width)
            N = math.ceil(env_length)
            hm = np.zeros((M, N), dtype=np.float32)

            scale_x = 20
            scale_y = 20
            octaves = 4  # np.random.randint(1, 5)
            persistence = 1
            lacunarity = 2

            for i in range(M):
                for j in range(N):
                    for o in range(octaves):
                        sx = scale_x * (1 / (lacunarity ** o))
                        sy = scale_y * (1 / (lacunarity ** o))
                        amp = persistence ** o
                        hm[i][j] += oSim.noise2d(i / sx, j / sy) * amp

            wmin, wmax = hm.min(), hm.max()
            hm = (hm - wmin) / (wmax - wmin) * height
            hm += current_height

        return hm, current_height

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

        contacts = [ctct_leg_1, ctct_leg_2, ctct_leg_3, ctct_leg_4, ctct_leg_5, ctct_leg_6]

        # Joints
        obs = p.getJointStates(self.robot, range(18), physicsClientId=self.client_ID) # pos, vel, reaction(6), prev_torque
        joint_angles = []
        joint_velocities = []
        joint_torques = []
        for o in obs:
            joint_angles.append(o[0])
            joint_velocities.append(o[1])
            joint_torques.append(o[3])
        return torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts

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
        scaled_action = self.scale_action(ctrl)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(18),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=scaled_action,
                                    forces=[self.max_joint_force] * 18,
                                    positionGains=[0.003] * 18,
                                    velocityGains=[0.03] * 18,
                                    physicsClientId=self.client_ID)

        for i in range(self.sim_steps_per_iter):
            p.stepSimulation(physicsClientId=self.client_ID)
            if self.animate or render: time.sleep(0.004)



        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts = self.get_obs()
        xd, yd, zd = torso_vel
        qx, qy, qz, qw = torso_quat

        # Reward shaping
        torque_pen = np.mean(np.square(joint_torques))

        self.xd_queue.append(xd)
        if len(self.xd_queue) > 12:
            self.xd_queue.pop(0)
        xd_av = sum(self.xd_queue) / len(self.xd_queue)

        velocity_rew = 1. / (abs(xd_av - self.target_vel) + 1.) - 1. / (self.target_vel + 1.)
        velocity_rew *= (0.3 / self.target_vel)

        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)
        q_yaw = 2 * np.arccos(qw)

        r_neg = np.square(q_yaw) * 0.5 + \
                np.square(pitch) * 0.05 + \
                np.square(roll) * 0.05 + \
                torque_pen * 0.000 + \
                np.square(zd) * 0.05
        r_pos = velocity_rew * 7
        r = np.clip(r_pos - r_neg, -3, 3)

        scaled_joint_angles = self.scale_joints(joint_angles)
        env_obs = np.concatenate((scaled_joint_angles, torso_quat, contacts))

        if self.step_counter:
            env_obs = np.concatenate((env_obs, [self.step_encoding]))

        self.step_ctr += 1
        self.step_encoding = (float(self.step_ctr) / self.max_steps) * 2 - 1
        done = self.step_ctr > self.max_steps or np.abs(roll) > 1.57 or np.abs(pitch) > 1.57

        if np.random.rand() < 0.0:
            self.make_heightfield()

        return env_obs, r, done, {}

    def reset(self):
        self.step_ctr = 0
        self.step_encoding = (float(self.step_ctr) / self.max_steps) * 2 - 1
        # p.changeDynamics(self.robot, linkIndex=-1, lateralFriction=1)
        # p.changeDynamics(self.robot, linkIndex=3, lateralFriction=1)

        if np.random.rand() < self.env_change_prob:
            self.generate_hybrid_env(self.n_envs, self.max_steps)

        joint_init_pos_list = self.scale_action([0] * 18)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(18)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, .25], [0, 0, 0, 1], physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(18),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * 18,
                                    forces=[self.max_joint_force] * 18,
                                    physicsClientId=self.client_ID)
        for i in range(10):
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

    def demo(self):
        self.reset()
        n_rep = 600
        force = 2
        for i in range(100):
            for j in range(n_rep):
                p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                            jointIndices=range(18),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=[0.5] * 18,
                                            forces=[force] * 18,
                                            physicsClientId=self.client_ID)

                p.stepSimulation(physicsClientId=self.client_ID)
                time.sleep(0.004)

            for j in range(n_rep):
                p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                            jointIndices=range(18),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=[-0.5] * 18,
                                            forces=[force] * 18,
                                            physicsClientId=self.client_ID)

                p.stepSimulation(physicsClientId=self.client_ID)
                time.sleep(0.004)

            for j in range(n_rep):
                p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                            jointIndices=range(18),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=[0,2,-2] * 6,
                                            forces=[force] * 18,
                                            physicsClientId=self.client_ID)
                p.stepSimulation(physicsClientId=self.client_ID)
                time.sleep(0.004)

            for j in range(n_rep):
                p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                            jointIndices=range(18),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=[0,-2,2] * 6,
                                            forces=[force] * 18,
                                            physicsClientId=self.client_ID)

                p.stepSimulation(physicsClientId=self.client_ID)
                time.sleep(0.004)

    def demo_step(self):
        self.reset()
        n_rep = 30
        for i in range(100):
            for j in range(n_rep):
                self.step([0,0,0] * 6)

            for j in range(n_rep):
                self.step([0,-1,-1] * 6)

            for j in range(n_rep):
                self.step([0,1,1] * 6)

            for j in range(n_rep):
                self.step([1,0,0] * 6)

            for j in range(n_rep):
                self.step([-1,0,0] * 6)

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

if __name__ == "__main__":
    env = HexapodBulletEnv(animate=True)
    env.demo_step()

    # TODO: Fix hexapod simulation forces and speed and compare to real platform to have correct everything