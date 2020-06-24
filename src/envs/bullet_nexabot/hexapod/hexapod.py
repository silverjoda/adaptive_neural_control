import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import torch as T
import gym
from gym import spaces

class HexapodBulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, animate=False, max_steps=200, seed=None):
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

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hexapod.urdf"), physicsClientId=self.client_ID)
        self.terrain = self.make_heightfield()

        self.obs_dim = 28
        self.act_dim = 18

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        self.joints_rads_low = np.array([-0.3, -1.6, 0.7] * 6)
        self.joints_rads_high = np.array([0.3, 0.0, 1.9] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.max_joint_force = 1.4
        self.target_vel = 0.4
        self.sim_steps_per_iter = 10
        self.step_ctr = 0
        self.xd_queue = []

    def make_heightfield(self):
        if hasattr(self, 'terrain'):
            p.removeBody(self.terrain, self.client_ID)

        heightPerturbationRange = 0.00
        numHeightfieldRows = 256
        numHeightfieldColumns = 256
        heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns
        for j in range(int(numHeightfieldColumns / 2)):
            for i in range(int(numHeightfieldRows / 2)):
                height = random.uniform(0, heightPerturbationRange)
                heightfieldData[2 * i + 2 * j * numHeightfieldRows] = height
                heightfieldData[2 * i + 1 + 2 * j * numHeightfieldRows] = height
                heightfieldData[2 * i + (2 * j + 1) * numHeightfieldRows] = height
                heightfieldData[2 * i + 1 + (2 * j + 1) * numHeightfieldRows] = height

        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.05, .05, 1],
                                              heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
                                              heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows,
                                              numHeightfieldColumns=numHeightfieldColumns)
        terrain = p.createMultiBody(0, terrainShape)
        p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_ID)
        return terrain

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
                                    positionGains=[0.01] * 18,
                                    velocityGains=[0.07] * 18,
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

        self.step_ctr += 1
        done = self.step_ctr > self.max_steps or np.abs(roll) > 1.57 or np.abs(pitch) > 1.57

        if np.random.rand() < 0.0:
            self.make_heightfield()

        return env_obs, r, done, {}

    def reset(self):
        self.step_ctr = 0
        # p.changeDynamics(self.robot, linkIndex=-1, lateralFriction=1)
        # p.changeDynamics(self.robot, linkIndex=3, lateralFriction=1)

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
        n_rep = 50
        for i in range(100):
            for j in range(n_rep):
                self.step([0,0,0] * 6, render=True)

            for j in range(n_rep):
                self.step([0,-1,-1] * 6, render=True)

            for j in range(n_rep):
                self.step([0,1,1] * 6, render=True)

            for j in range(n_rep):
                self.step([1,0,0] * 6, render=True)

            for j in range(n_rep):
                self.step([-1,0,0] * 6, render=True)

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

if __name__ == "__main__":
    env = HexapodBulletEnv(animate=True)
    env.demo_step()