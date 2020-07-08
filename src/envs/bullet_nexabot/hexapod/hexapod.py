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

# To mirror quaternion along x-z plane (or y axis) just use q_mirror = [qx, -qy, qz, -qw]

class HexapodBulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, animate=False, max_steps=200, seed=None, step_counter=False, terrain_name=None, training_mode="straight"):
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
        self.replace_envs = False
        self.n_envs = 1
        self.env_width = 60
        self.env_length = self.max_steps
        self.env_change_prob = 0.1
        self.walls = False
        self.mesh_scale_lat = 0.1
        self.mesh_scale_vert = 2

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hexapod.urdf"), physicsClientId=self.client_ID)
        self.generate_rnd_env()

        self.obs_dim = 18 + 6 + 4 + int(step_counter)
        self.act_dim = 18

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        self.joints_rads_low = np.array([-0.3, -1.6, 0.7] * 6)
        self.joints_rads_high = np.array([0.3, 0.0, 1.9] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.max_joint_force = 1.4
        self.target_vel = 0.2
        self.sim_steps_per_iter = 24
        self.step_ctr = 0
        self.xd_queue = []

    def make_heightfield(self, height_map=None):
        if hasattr(self, 'terrain'):
            p.removeBody(self.terrain, self.client_ID)
        if height_map is None:
            heightfieldData = np.zeros(self.env_width*self.env_length)
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
                                                  numHeightfieldColumns=height_map.shape[1])
        terrain = p.createMultiBody(0, terrainShape)
        p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_ID)
        return terrain

    def generate_heightmap(self, env_name):
        current_height = 0
        if env_name == "flat" or env_name is None:
            hm = np.ones((self.env_length, self.env_width)) * current_height

        if env_name == "tiles":
            sf = 4
            hm = np.random.randint(0, 18,
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
            stair_height = 12
            stair_width = 2

            initial_offset = self.env_length // 2 - self.env_length // 8
            n_steps = min(math.floor(self.env_length / stair_width) - 1, 10)

            for i in range(n_steps):
                hm[initial_offset + i * stair_width: initial_offset + i * stair_width + stair_width, :] = current_height
                current_height += stair_height

            hm[n_steps * stair_width + initial_offset:, :] = current_height

        if env_name == "stairs_down":
            stair_height = 12
            stair_width = 2

            initial_offset = self.env_length // 2 - self.env_length // 8
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

            height = 50

            M = math.ceil(self.env_width)
            N = math.ceil(self.env_length)
            hm = np.zeros((N, M), dtype=np.float32)

            scale_x = 20
            scale_y = 20
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
        scaled_action = self.scale_action(ctrl_clipped)
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
            if (self.animate or render) and True: time.sleep(0.0038)

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
        q_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        if self.training_mode == "straight":
            r_neg = np.square(q_yaw) * 0.7 + \
                    np.square(pitch) * 0.1 + \
                    np.square(roll) * 0.1 + \
                    torque_pen * 0.00001 + \
                    np.square(zd) * 0.5
            r_pos = velocity_rew * 7
            r = np.clip(r_pos - r_neg, -3, 3)
        elif self.training_mode == "straight_rough":
            r_neg = np.square(q_yaw) * 0.5 + \
                    np.square(pitch) * 0.1 + \
                    np.square(roll) * 0.1 + \
                    torque_pen * 0.000 + \
                    np.square(zd) * 0.1
            r_pos = velocity_rew * 7
            r = np.clip(r_pos - r_neg, -3, 3)
        elif self.training_mode == "turn_left":
            r_neg = np.square(xd) * 0.3 + np.square(yd) * 0.3
            r_pos = torso_angular_vel[2] * 7
            r = np.clip(r_pos - r_neg, -3, 3)
        elif self.training_mode == "turn_right":
            r_neg = np.square(xd) * 0.3 + np.square(yd) * 0.3
            r_pos = -torso_angular_vel[2] * 7
            r = np.clip(r_pos - r_neg, -3, 3)
        elif self.training_mode == "stairs":
            r_neg = np.square(q_yaw) * 0.5 + \
                    np.square(pitch) * 0.0 + \
                    np.square(roll) * 0.0 + \
                    torque_pen * 0.00001 + \
                    np.square(zd) * 0.0
            velocity_rew = 1. / (abs(xd_av - self.target_vel * 0.6) + 1.) - 1. / (self.target_vel * 0.6 + 1.)
            velocity_rew *= (0.3 / (self.target_vel * 0.6))
            r_pos = velocity_rew * 7
            r = np.clip(r_pos - r_neg, -3, 3)
        else:
            print("No mode selected")
            exit()

        # TODO: Try reward which forces equal work with all legs
        # TODO: Make motor penalty which penalizes work, not torque

        scaled_joint_angles = self.scale_joints(joint_angles)
        env_obs = np.concatenate((scaled_joint_angles, torso_quat, contacts))

        if self.step_counter:
            env_obs = np.concatenate((env_obs, [self.step_encoding]))

        self.step_ctr += 1
        self.step_encoding = (float(self.step_ctr) / self.max_steps) * 2 - 1
        done = self.step_ctr > self.max_steps or np.abs(roll) > 1.57 or np.abs(pitch) > 1.57

        return env_obs, r, done, {}

    def reset(self):
        self.step_ctr = 0
        self.step_encoding = (float(self.step_ctr) / self.max_steps) * 2 - 1
        # p.changeDynamics(self.robot, linkIndex=-1, lateralFriction=1)
        # p.changeDynamics(self.robot, linkIndex=3, lateralFriction=1)

        if np.random.rand() < self.env_change_prob:
            self.generate_rnd_env()

        # Get heightmap height at robot position
        if self.terrain is None:
            spawn_height = 0
        else:
            spawn_height = 0.5 * np.max(self.terrain_hm[self.env_length // 2 - 3:self.env_length // 2 + 3, self.env_width // 2 - 3 : self.env_width // 2 + 3]) * self.mesh_scale_vert

        joint_init_pos_list = self.scale_action([0] * 18)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(18)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, spawn_height + 0.15], [0, 0, 0, 1], physicsClientId=self.client_ID)
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

        # for i in range(1000):
        #     p.resetBasePositionAndOrientation(self.robot, [0, 0, .25], [0.0, 0.16, -0.39, 0.91], physicsClientId=self.client_ID) # x,y,z,w
        #     time.sleep(0.004)
        #
        # for i in range(1000):
        #     p.resetBasePositionAndOrientation(self.robot, [0, 0, .25], [0.0, -0.16, -0.39, -0.91], physicsClientId=self.client_ID)
        #     time.sleep(0.004)
        # exit()

        self.reset()
        n_rep = 30
        for i in range(100):
            for j in range(n_rep):
                obs, _, _, _ = self.step([0,0,0] * 6)
            print(obs[:18])

            for j in range(n_rep):
                obs, _, _, _ = self.step([0,-1,-1] * 6)
            print(obs[:18])

            for j in range(n_rep):
                obs, _, _, _ = self.step([0,1,1] * 6)
            print(obs[:18])

            for j in range(n_rep):
                obs, _, _, _ = self.step([1,0,0] * 6)
            print(obs[:18])

            for j in range(n_rep):
                obs, _, _, _ = self.step([-1,0,0] * 6)
            print(obs[:18])

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

if __name__ == "__main__":
    env = HexapodBulletEnv(animate=True)
    env.demo_step()

    # TODO: Fix hexapod simulation forces and speed and compare to real platform to have correct everything