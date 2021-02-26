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

class HexapodBulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, config):
        self.config = config
        self.seed = self.config["seed"]

        if self.seed is not None:
            self.set_seed(self.seed, self.seed)
            print(f"Setting from given seed: {self.seed}")
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
        self.obs_dim = 27 + 36 * self.config["velocities_and_torques"]
        self.act_dim = 18
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        self.joints_rads_low = np.array(config["joints_rads_low"] * 6)
        self.joints_rads_high = np.array(config["joints_rads_high"] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

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

        self.create_targets()

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
            self.target = np.array(self.config["target_spawn_mu"]) + np.random.rand(2) * np.array(
                self.config["target_spawn_sigma"]) - np.array(self.config[
                                                                  "target_spawn_sigma"]) / 2

            self.target_body = p.createMultiBody(baseMass=0,
                                                 baseVisualShapeIndex=self.target_visualshape,
                                                 basePosition=[self.target[0], self.target[1], 0],
                                                 physicsClientId=self.client_ID)
        else:
            self.target = np.array(self.config["target_spawn_mu"]) + np.random.rand(2) * np.array(
                self.config["target_spawn_sigma"]) - np.array(self.config["target_spawn_sigma"]) / 2
            p.resetBasePositionAndOrientation(self.target_body, [self.target[0], self.target[1], 0], [0, 0, 0, 1],
                                              physicsClientId=self.client_ID)

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

        if env_name == "perlin":
            oSim = OpenSimplex(seed=int(time.time()))

            height = self.config["perlin_height"]

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

        if env_name == "obstacle":
            oSim = OpenSimplex(seed=int(time.time()))

            height = self.config["obstacle_height"] * self.config["training_difficulty"] # 30-40

            M = math.ceil(self.config["env_width"])
            N = math.ceil(self.config["env_length"])
            hm = np.zeros((N, M), dtype=np.float32)

            scale_x = 15
            scale_y = 15
            octaves = 4 # np.random.randint(1, 5)
            persistence = 1
            lacunarity = 2

            for i in range(int(N/2) + 3, int(N/2) + 7):
                for j in range(M):
                    for o in range(octaves):
                        sx = scale_x * (1 / (lacunarity ** o))
                        sy = scale_y * (1 / (lacunarity ** o))
                        amp = persistence ** o
                        hm[i][j] += abs(oSim.noise2d(i / sx, j / sy) * amp)

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
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                  meshScale=[self.config["mesh_scale_lat"],
                                                             self.config["mesh_scale_lat"],
                                                             self.config["mesh_scale_vert"]],
                                                  heightfieldTextureScaling=(self.config["env_width"] - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=self.config["max_steps"],
                                                  numHeightfieldColumns=self.config["env_width"],
                                                  physicsClientId=self.client_ID)
        else:
            heightfieldData = height_map.ravel(order='F')
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                  meshScale=[self.config["mesh_scale_lat"],
                                                             self.config["mesh_scale_lat"],
                                                             self.config["mesh_scale_vert"]],
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
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)  # xyz and quat: x,y,z,w
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)

        contacts = [int(len(p.getContactPoints(self.robot, self.terrain, i * 3 + 2, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1for i in range(6)]
        ctct_torso = int(len(p.getContactPoints(self.robot, self.terrain, -1, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1

        # Joints
        obs = p.getJointStates(self.robot, range(18), physicsClientId=self.client_ID)  # pos, vel, reaction(6), prev_torque
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
            self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.urdf_name),
                                    physicsClientId=self.client_ID)

        # Randomize robot params
        self.randomized_params = {"mass": self.config["mass"] ,
                                  "lateral_friction": self.config["lateral_friction"],
                                  "max_joint_force": self.config["max_joint_force"],
                                  "actuator_position_gain": self.config["actuator_position_gain"],
                                  "actuator_velocity_gain": self.config["actuator_velocity_gain"],
                                  "max_actuator_velocity": self.config["max_actuator_velocity"]}

        p.changeDynamics(self.robot, -1, mass=self.randomized_params["mass"],
                         physicsClientId=self.client_ID)
        p.changeDynamics(self.robot, -1, lateralFriction=self.randomized_params["lateral_friction"],
                         physicsClientId=self.client_ID)
        for i in range(6):
            p.changeDynamics(self.robot, 3 * i + 2, lateralFriction=self.randomized_params["lateral_friction"],
                             physicsClientId=self.client_ID)

        return self.robot

    def step(self, ctrl_raw, render=False):
        scaled_action = self.norm_to_rads(ctrl_raw)

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

        # p.setJointMotorControlArray(bodyUniqueId=self.robot,
        #                             jointIndices=range(18),
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPositions=scaled_action,
        #                             forces=[self.config["max_joint_force"]] * 18,
        #                             physicsClientId=self.client_ID)

        for i in range(self.config["sim_steps_per_iter"]):
            p.stepSimulation(physicsClientId=self.client_ID)
            if (self.config["animate"] or render) and True: time.sleep(self.config["sim_step"])

        self.step_ctr += 1

        # Get all observations
        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso = self.get_obs()
        xd, yd, zd = torso_vel
        thd, phid, psid = torso_angular_vel

        scaled_joint_angles = self.rads_to_norm(joint_angles)

        # Calculate yaw
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        # Orientation angle
        tar_angle = np.arctan2(self.target[1] - torso_pos[1], self.target[0] - torso_pos[0])
        yaw_deviation = np.min((abs((yaw % 6.283) - (tar_angle % 6.283)), abs(yaw - tar_angle)))
        signed_deviation = yaw - tar_angle

        # Check if the agent has reached a target
        target_dist = np.sqrt((torso_pos[0] - self.target[0]) ** 2 + (torso_pos[1] - self.target[1]) ** 2)

        r = (self.prev_target_dist - target_dist) * 50 - abs(zd) * 0.2 - abs(phid) * 0.02 - abs(thd) * 0.02

        if target_dist < self.config["target_proximity_threshold"] or (np.abs(torso_pos[0]) > self.target[0]):
            reached_target = True
            self.update_targets()
            self.prev_target_dist = np.sqrt(
                (torso_pos[0] - self.target[0]) ** 2 + (torso_pos[1] - self.target[1]) ** 2)
            tar_angle = np.arctan2(self.target[1] - torso_pos[1], self.target[0] - - torso_pos[0])
            yaw_deviation = np.min(
                (abs((yaw % 6.283) - (tar_angle % 6.283)), abs(yaw - tar_angle)))
        else:
            reached_target = False
            self.prev_target_dist = target_dist

        # Assemble agent observation
        time_feature = [(float(self.step_ctr) / self.config["max_steps"]) * 2 - 1]
        if self.obs_dim > 33:
            compiled_obs = torso_quat, torso_vel, [signed_deviation], time_feature, scaled_joint_angles, joint_torques, joint_velocities
        else:
            compiled_obs = torso_quat, torso_vel, [signed_deviation], time_feature, scaled_joint_angles
        compiled_obs_flat = [item for sublist in compiled_obs for item in sublist]
        env_obs = np.array(compiled_obs_flat).astype(np.float32)

        done = self.step_ctr > self.config["max_steps"] or reached_target

        if np.abs(roll) > 1.57 or np.abs(pitch) > 1.57:
            print("WARNING!! Absolute roll and pitch values exceed bounds: roll: {}, pitch: {}".format(roll, pitch))
            done = True

        if abs(torso_pos[0]) > 6 or abs(torso_pos[1]) > 6 or abs(torso_pos[2]) > 2.5:
            print("WARNING: TORSO OUT OF RANGE!!")
            done = True

        return env_obs, r, done, {}

    def reset(self, force_randomize=None):
        if hasattr(self, 'terrain'):
            p.removeBody(self.terrain, physicsClientId=self.client_ID)
        if hasattr(self, 'robot'):
            p.removeBody(self.robot, physicsClientId=self.client_ID)

        del self.robot
        del self.terrain
        del self.target_body
        self.target = None

        p.resetSimulation(physicsClientId=self.client_ID)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        self.robot = self.load_robot()
        if not self.config["terrain_name"] == "flat":
            self.terrain = self.generate_rnd_env()
        else:
            self.terrain = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)
        self.create_targets()

        # Reset episodal vars
        self.step_ctr = 0
        self.episode_ctr += 1
        self.prev_yaw_dev = 0

        # Get heightmap height at robot position
        if self.terrain is None or self.config["terrain_name"] == "flat":
            spawn_height = 0
        else:
            spawn_height = 0.5 * np.max(
                self.terrain_hm[self.config["env_length"] // 2 - 3:self.config["env_length"] // 2 + 3,
                self.config["env_width"] // 2 - 3: self.config["env_width"] // 2 + 3]) * self.config["mesh_scale_vert"]

        # Random initial rotation
        rnd_rot = np.random.rand() * 0.3 - 0.15
        rnd_quat = p.getQuaternionFromAxisAngle([0, 0, 1], rnd_rot)

        joint_init_pos_list = self.norm_to_rads([0] * 18)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(18)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, spawn_height + 0.15], rnd_quat,
                                          physicsClientId=self.client_ID)
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
                        [p.resetJointState(self.robot, k, joint_angles_prev[k], 0, physicsClientId=self.client_ID) for k
                         in range(18)]
                        p.stepSimulation(physicsClientId=self.client_ID)
                        time.sleep(0.6)

                        [p.resetJointState(self.robot, k, joint_angles_desired[k], 0, physicsClientId=self.client_ID)
                         for k in range(18)]
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
        n_steps = 200
        VERBOSE = True
        while True:
            t1 = time.time()
            sc = 1.0
            test_acts = [[0, 0, 0], [0, sc, sc], [0, -sc, -sc], [0, sc, -sc], [0, -sc, sc], [sc, 0, 0], [-sc, 0, 0]]
            for i, a in enumerate(test_acts):
                for j in range(n_steps):
                    # a = list(np.random.randn(3))
                    scaled_obs, _, _, _ = self.step(a * 6)
                _, _, _, _, joint_angles, _, joint_torques, contacts, ctct_torso = self.get_obs()
                if VERBOSE:
                    print("Obs rads: ", joint_angles)
                    print("Obs normed: ", self.rads_to_norm(joint_angles))
                    print("For action rads: ", self.norm_to_rads(a * 6))
                    print("action normed: ", a)
                    # input()
                # self.reset()

            t2 = time.time()
            print("Time taken for iteration: {}".format(t2 - t1))

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)


if __name__ == "__main__":
    import yaml

    with open("configs/hexapod.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = HexapodBulletEnv(env_config)
    env.test_leg_coordination()
