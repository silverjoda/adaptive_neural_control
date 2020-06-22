import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import torch as T
import gym
from gym import spaces

class QuadrupedBulletEnv(gym.Env):
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

        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "quadruped.urdf"), physicsClientId=self.client_ID)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        self.obs_dim = 20
        self.act_dim = 12

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        self.joints_rads_low = np.array([-0.3, -1.8, 1.4] * 4)
        self.joints_rads_high = np.array([0.3, -0.2, 2.6] * 4)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.max_joint_force = 1.4
        self.target_vel = 0.15
        self.sim_steps_per_iter = 10
        self.step_ctr = 0
        self.xd_queue = []

    def get_obs(self):
        # Torso
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID) # xyz and quat: x,y,z,w
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)

        ctct_leg_1 = int(len(p.getContactPoints(self.robot, self.plane, 2, -1, physicsClientId=self.client_ID)) > 0)
        ctct_leg_2 = int(len(p.getContactPoints(self.robot, self.plane, 5, -1, physicsClientId=self.client_ID)) > 0)
        ctct_leg_3 = int(len(p.getContactPoints(self.robot, self.plane, 8, -1, physicsClientId=self.client_ID)) > 0)
        ctct_leg_4 = int(len(p.getContactPoints(self.robot, self.plane, 11, -1, physicsClientId=self.client_ID)) > 0)

        contacts = [ctct_leg_1, ctct_leg_2, ctct_leg_3, ctct_leg_4]

        # Joints
        obs = p.getJointStates(self.robot, range(12), physicsClientId=self.client_ID) # pos, vel, reaction(6), prev_torque
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
        time.sleep(0.0035 * self.sim_steps_per_iter)

    def step(self, ctrl, render=False):
        ctrl_clipped = np.clip(ctrl, -1, 1)
        scaled_action = self.scale_action(ctrl_clipped)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(12),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=scaled_action,
                                    forces=[self.max_joint_force] * 12,
                                    positionGains=[0.01] * 12,
                                    velocityGains=[0.07] * 12,
                                    physicsClientId=self.client_ID)

        for i in range(self.sim_steps_per_iter):
            p.stepSimulation(physicsClientId=self.client_ID)
            if render: time.sleep(0.004)

        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts = self.get_obs()
        xd, yd, zd = torso_vel
        qx, qy, qz, qw = torso_quat

        # Reward shaping
        torque_pen = np.mean(np.square(joint_torques))

        self.xd_queue.append(xd)
        if len(self.xd_queue) > 15:
            self.xd_queue.pop(0)
        xd_av = sum(self.xd_queue) / len(self.xd_queue)

        velocity_rew = 1. / (abs(xd_av - self.target_vel) + 1.) - 1. / (self.target_vel + 1.)
        velocity_rew *= (0.3 / self.target_vel)

        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)
        q_yaw = 2 * np.arccos(qw)

        r_neg = np.square(q_yaw) * 0.9 + \
                np.square(pitch) * 0.5 + \
                np.square(roll) * 0.5 + \
                torque_pen * 0.000 + \
                np.square(zd) * 0.5
        r_pos = velocity_rew * 7
        r = np.clip(r_pos - r_neg, -3, 3)

        scaled_joint_angles = self.scale_joints(joint_angles)
        env_obs = np.concatenate((scaled_joint_angles, torso_quat, contacts))

        self.step_ctr += 1
        done = self.step_ctr > self.max_steps or np.abs(roll) > 1.57 or np.abs(pitch) > 1.57

        return env_obs, r, done, {}

    def reset(self):
        self.step_ctr = 0
        # p.changeDynamics(self.robot, linkIndex=-1, lateralFriction=1)
        # p.changeDynamics(self.robot, linkIndex=3, lateralFriction=1)

        joint_init_pos_list = self.scale_action([0] * 12)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(12)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, .15], [0, 0, 0, 1], physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(12),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * 12,
                                    forces=[self.max_joint_force] * 12,
                                    physicsClientId=self.client_ID)
        for i in range(10):
            p.stepSimulation(physicsClientId=self.client_ID)

        obs, _, _, _ = self.step(np.zeros(self.act_dim))
        return obs

    def test(self, policy, slow=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.render_prob = 1.0
        total_rew = 0
        for i in range(100):
            obs = self.reset()
            cr = 0
            for j in range(self.max_steps):
                nn_obs = T.FloatTensor(obs.astype(np.float32)).unsqueeze(0)
                action = policy(nn_obs).detach()
                obs, r, done, od, = self.step(action[0].numpy())
                cr += r
                total_rew += r
                if slow:
                    time.sleep(0.001)
            print("Total episode reward: {}".format(cr))
        print("Total reward: {}".format(total_rew))

    def demo(self):
        self.reset()
        n_rep = 600
        force = 2
        for i in range(100):
            for j in range(n_rep):
                p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                            jointIndices=range(12),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=[0.5] * 12,
                                            forces=[force] * 12,
                                            physicsClientId=self.client_ID)

                p.stepSimulation(physicsClientId=self.client_ID)
                time.sleep(0.004)

            for j in range(n_rep):
                p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                            jointIndices=range(12),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=[-0.5] * 12,
                                            forces=[force] * 12,
                                            physicsClientId=self.client_ID)

                p.stepSimulation(physicsClientId=self.client_ID)
                time.sleep(0.004)

            for j in range(n_rep):
                p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                            jointIndices=range(12),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=[0,2,-2] * 4,
                                            forces=[force] * 12,
                                            physicsClientId=self.client_ID)
                p.stepSimulation(physicsClientId=self.client_ID)
                time.sleep(0.004)

            for j in range(n_rep):
                p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                            jointIndices=range(12),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=[0,-2,2] * 4,
                                            forces=[force] * 12,
                                            physicsClientId=self.client_ID)

                p.stepSimulation(physicsClientId=self.client_ID)
                time.sleep(0.004)

    def demo_step(self):
        self.reset()
        n_rep = 100
        for i in range(100):
            for j in range(n_rep):
                self.step([0,0,0] * 4, render=True)

            for j in range(n_rep):
                self.step([0,-1,-1] * 4, render=True)

            for j in range(n_rep):
                self.step([0,1,1] * 4, render=True)

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

if __name__ == "__main__":
    env = QuadrupedBulletEnv(animate=True)
    env.demo_step()