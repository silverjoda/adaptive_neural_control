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

        if (self.config["animate"]):
            self.client_ID = p.connect(p.GUI)
            print(" --Starting GUI mode-- ")
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        self.obs_dim = 20
        self.act_dim = 12

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        self.joints_rads_low = np.array(config["joints_rads_low"] * 4)
        self.joints_rads_high = np.array(config["joints_rads_high"] * 4)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "quadruped.urdf"),
                                physicsClientId=self.client_ID)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

        for i in range(4):
            p.changeDynamics(self.robot, 3 * i + 2, lateralFriction=config["lateral_friction"])
        p.changeDynamics(self.robot, -1, lateralFriction=config["lateral_friction"])

        self.step_ctr = 0
        self.xd_queue = []

    def get_obs(self):
        # Torso
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID) # xyz and quat: x,y,z,w
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)

        ctct_leg_1 = int(len(p.getContactPoints(self.robot, self.plane, 2, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        ctct_leg_2 = int(len(p.getContactPoints(self.robot, self.plane, 5, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        ctct_leg_3 = int(len(p.getContactPoints(self.robot, self.plane, 8, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
        ctct_leg_4 = int(len(p.getContactPoints(self.robot, self.plane, 11, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1

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

    def rads_to_norm(self, joints):
        sjoints = np.array(joints)
        sjoints = ((sjoints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        return sjoints

    def norm_to_rads(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    def load_robot(self):
        # Remove old robot
        if self.robot is not None:
            p.removeBody(self.robot)

        # Randomize robot params
        self.robot_params = {"mass": 1 + np.random.rand() * 0.5,
                             "boom": 0.1 + np.random.rand() * 0.5,
                             "motor_inertia_coeff": 0.7 + np.random.rand() * 0.25,
                             "motor_force_multiplier": 60 + np.random.rand() * 30}

        # Write params to URDF file
        with open(self.config["urdf_name"], "r") as in_file:
            buf = in_file.readlines()

        index = self.config["urdf_name"].find('.urdf')
        output_urdf = self.config["urdf_name"][:index] + '_rnd' + self.config["urdf_name"][index:]

        # Change link lengths in urdf
        with open(output_urdf, "w") as out_file:
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
        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_urdf), physicsClientId=self.client_ID)

        # Change base mass
        p.changeDynamics(self.robot, -1, mass=self.robot_params["mass"])

    def render(self, close=False):
        pass

    def step(self, ctrl, render=False):
        ctrl_clipped = np.clip(ctrl, -1, 1)
        scaled_action = self.norm_to_rads(ctrl_clipped)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(12),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=scaled_action,
                                    forces=[self.config["max_joint_force"]] * 12,
                                    positionGains=[0.01] * 12,
                                    velocityGains=[0.07] * 12,
                                    physicsClientId=self.client_ID)

        p.stepSimulation(physicsClientId=self.client_ID)
        if self.config["animate"] or render: time.sleep(0.004)

        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts = self.get_obs()
        xd, yd, zd = torso_vel
        qx, qy, qz, qw = torso_quat

        # Reward shaping
        torque_pen = np.mean(np.square(joint_torques))

        self.xd_queue.append(xd)
        if len(self.xd_queue) > 10:
            self.xd_queue.pop(0)
        xd_av = sum(self.xd_queue) / len(self.xd_queue)

        velocity_rew = 1. / (abs(xd_av - self.config["target_vel"]) + 1.) - 1. / (self.config["target_vel"] + 1.)
        velocity_rew *= (0.3 / self.config["target_vel"])

        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)
        q_yaw = 2 * np.arccos(qw)

        r_neg = np.square(q_yaw) * 0.5 + \
                np.square(pitch) * 0.05 + \
                np.square(roll) * 0.05 + \
                torque_pen * 0.000 + \
                np.square(zd) * 0.05
        r_pos = velocity_rew * 7
        r = np.clip(r_pos - r_neg, -3, 3)

        scaled_joint_angles = self.rads_to_norm(joint_angles)
        env_obs = np.concatenate((scaled_joint_angles, torso_quat, contacts))

        self.step_ctr += 1
        done = self.step_ctr > self.config["max_steps"]or np.abs(roll) > 1.57 or np.abs(pitch) > 1.57

        return env_obs, r, done, {}

    def reset(self, randomize=False):
        self.step_ctr = 0

        if self.config["is_variable"] or randomize:
            self.load_robot()

        joint_init_pos_list = self.norm_to_rads([0] * 12)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(12)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, .15], [0, 0, 0, 1], physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(12),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * 12,
                                    forces=[self.config["max_joint_force"]] * 12,
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
            for j in range(self.config["max_steps"]):
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
        n_steps = 400
        VERBOSE = True
        while True:
            t1 = time.time()
            sc = 1.0
            test_acts = [[0, 0, 0], [0, sc, sc], [0, -sc, -sc], [0, sc, -sc], [0, -sc, sc], [sc, 0, 0], [-sc, 0, 0]]
            for i, a in enumerate(test_acts):
                for j in range(n_steps):
                    # a = list(np.random.randn(3))
                    scaled_obs, _, _, _ = self.step(a * 4)
                _, _, _, _, joint_angles, joint_velocities, joint_torques, contacts = self.get_obs()
                if VERBOSE:
                    print("Obs rads: ", joint_angles)
                    print("Obs normed: ", self.rads_to_norm(joint_angles))
                    print("For action rads: ", self.norm_to_rads(a * 4))
                    print("action normed: ", a)
                    # input()

                self.reset()

            t2 = time.time()
            print("Time taken for iteration: {}".format(t2 - t1))

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

if __name__ == "__main__":
    import yaml
    with open("../../../algos/SB/configs/quadruped_config.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = QuadrupedBulletEnv(env_config)
    env.test_leg_coordination()