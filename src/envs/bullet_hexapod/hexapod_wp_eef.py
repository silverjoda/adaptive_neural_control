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
import numpy as np

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
            self.seed = int((time.time() % 1) * 10000000)
            self.set_seed(self.seed, self.seed + 1)

        if (self.config["animate"]):
            self.client_ID = p.connect(p.GUI)
            print(" --Starting GUI mode-- ")
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        # Environment parameters
        self.act_dim = 14  # x_mult, y_offset, z_mult, z_offset, phase_offset, phase_0 ... phase_5
        self.obs_dim = 33

        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.step_ctr = 0
        self.angle = 0
        self.episode_ctr = 0

        self.urdf_name = config["urdf_name"]
        self.robot = self.load_robot()

        if config["terrain_name"] == "flat":
            self.terrain = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)
        else:
            self.generate_rnd_env()


        self.phases_op = np.array([-0.45378223061561584, 2.0218961238861084, 2.984712839126587, -1.3525876998901367, -1.7097233533859253, 1.027929663658142])
        self.x_mult, self.y_offset, self.z_mult, self.z_offset= [
            0.06,
            0.12,
            0.03,
            -0.12]

        self.create_targets()


    def set_seed(self, np_seed, T_seed):
        np.random.seed(np_seed)
        T.manual_seed(T_seed)

    def _seed(self, seed):
        np.random.seed(seed)
        T.manual_seed(seed)

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

        self.make_heightfield(self.terrain_hm)

        return

    def generate_heightmap(self, env_name):
        current_height = 0
        if env_name == "flat" or env_name is None:
            hm = np.ones((self.config["env_length"], self.config["env_width"])) * current_height

        if env_name == "tiles":
            sf = 3
            hm = np.random.randint(0, 20 * self.config["training_difficulty"],  # 15
                                   size=(self.config["env_length"] // sf, self.config["env_width"] // sf)).repeat(sf,
                                                                                                                  axis=0).repeat(
                sf, axis=1)
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
            hm[initial_offset:initial_offset + self.config["env_length"] // 4, :] = np.tile(
                np.linspace(current_height, current_height + max_height, self.config["env_length"] // 4)[:, np.newaxis],
                (1, self.config["env_width"]))
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
            oSim = OpenSimplex(seed=(self.seed + self.episode_ctr * (self.seed % 100)))

            height = self.config["perlin_height"] * self.config["training_difficulty"]  # 30-40

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
            self.terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
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
            self.terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                       meshScale=[self.config["mesh_scale_lat"],
                                                                  self.config["mesh_scale_lat"],
                                                                  self.config["mesh_scale_vert"]],
                                                       heightfieldTextureScaling=(self.config["env_width"] - 1) / 2,
                                                       heightfieldData=heightfieldData,
                                                       numHeightfieldRows=height_map.shape[0],
                                                       numHeightfieldColumns=height_map.shape[1],
                                                       physicsClientId=self.client_ID)
        self.terrain = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=self.terrainShape,
                                         physicsClientId=self.client_ID)

        p.resetBasePositionAndOrientation(self.terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_ID)

    def get_obs(self):
        # Torso
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot,
                                                                physicsClientId=self.client_ID)  # xyz and quat: x,y,z,w
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)

        contacts = [int(len(
            p.getContactPoints(self.robot, self.terrain, i * 3 + 2, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1
                    for i in range(6)]
        ctct_torso = int(
            len(p.getContactPoints(self.robot, self.terrain, -1, -1, physicsClientId=self.client_ID)) > 0) * 2 - 1

        # Joints
        obs = p.getJointStates(self.robot, range(18),
                               physicsClientId=self.client_ID)  # pos, vel, reaction(6), prev_torque
        joint_angles = []
        joint_velocities = []
        joint_torques = []
        for o in obs:
            joint_angles.append(o[0])
            joint_velocities.append(o[1])
            joint_torques.append(o[3])

        return torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso

    def render(self, close=False, mode=None):
        pass
        # if self.config["animate"]:
        #    time.sleep(self.config["sim_step"])

    def load_robot(self):
        # Remove old robot
        if not hasattr(self, 'robot'):
            self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.urdf_name),
                                    physicsClientId=self.client_ID)

            # Randomize robot params
            self.randomized_params = {"mass": self.config["mass"] + (np.random.rand() * 1.4 - 0.7) * self.config[
                "randomize_env"],
                                      "lateral_friction": self.config["lateral_friction"] + (
                                                  np.random.rand() * 1.2 - 0.6) *
                                                          self.config[
                                                              "randomize_env"],
                                      "max_joint_force": self.config["max_joint_force"] + (
                                                  np.random.rand() * 1.0 - 0.5) *
                                                         self.config[
                                                             "randomize_env"],
                                      "actuator_position_gain": self.config["actuator_position_gain"] + (
                                                  np.random.rand() * 0.4 - 0.2) * self.config[
                                                                    "randomize_env"],
                                      "actuator_velocity_gain": self.config["actuator_velocity_gain"] + (
                                                  np.random.rand() * 0.4 - 0.2) * self.config[
                                                                    "randomize_env"],
                                      "max_actuator_velocity": self.config["max_actuator_velocity"] + (
                                              np.random.rand() * 4.0 - 2.0) * self.config[
                                                                   "randomize_env"],
                                      }

        p.changeDynamics(self.robot, -1, mass=self.randomized_params["mass"],
                         physicsClientId=self.client_ID)
        p.changeDynamics(self.robot, -1, lateralFriction=self.randomized_params["lateral_friction"],
                         physicsClientId=self.client_ID)
        for i in range(6):
            p.changeDynamics(self.robot, 3 * i + 2, lateralFriction=self.randomized_params["lateral_friction"],
                             physicsClientId=self.client_ID)

        return self.robot

    def step(self, ctrl_raw, render=False):
        #current_phases = self.phases_op + np.tanh(ctrl_raw[0:6]) * np.pi * self.config["phase_scalar"]
        x_mult_arr = [self.x_mult + np.tanh(ctrl_raw[7]) * self.config["x_mult_scalar"], self.x_mult + np.tanh(ctrl_raw[8]) * self.config["x_mult_scalar"]] * 3

        targets = []
        for i in range(6):
            target_x = np.cos(-self.angle * 2 * np.pi + self.phases_op[i]) * x_mult_arr[i]
            target_y = self.y_offset
            target_z = np.clip(np.sin(-self.angle * 2 * np.pi + self.phases_op[i]) * self.z_mult + self.z_offset + np.tanh(ctrl_raw[8 + i]) * self.config["z_aux_scalar"], -0.13, -0.04)
            targets.append([target_x, target_y, target_z])

        joint_angles = self.my_ikt(targets, self.y_offset)

        self.angle += self.config["angle_increment"]

        for i in range(18):
            p.setJointMotorControl2(bodyUniqueId=self.robot,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_angles[i],
                                    force=self.randomized_params["max_joint_force"],
                                    positionGain=self.randomized_params["actuator_position_gain"],
                                    velocityGain=self.randomized_params["actuator_velocity_gain"],
                                    maxVelocity=self.randomized_params["max_actuator_velocity"],
                                    physicsClientId=self.client_ID)

        for i in range(self.config["sim_steps_per_iter"]):
            p.stepSimulation(physicsClientId=self.client_ID)
            if (self.config["animate"] or render) and True: time.sleep(0.00417)

        self.step_ctr += 1

        # Get all observations
        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso = self.get_obs()
        xd, yd, zd = torso_vel
        thd, phid, psid = torso_angular_vel

        # Calculate yaw
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat, physicsClientId=self.client_ID)

        # Orientation angle
        tar_angle = np.arctan2(self.target[1] - torso_pos[1], self.target[0] - torso_pos[0])
        yaw_deviation = np.min((abs((yaw % 6.283) - (tar_angle % 6.283)), abs(yaw - tar_angle)))
        signed_deviation = yaw - tar_angle

        # Compute heading reward
        # yaw_dev_diff = abs(self.prev_yaw_deviation) - abs(yaw_deviation)
        # yaw_dev_sign = np.sign(yaw_dev_diff)
        # heading_rew = np.minimum(np.abs(yaw_deviation), 3) * np.clip(yaw_dev_sign * np.square(yaw_dev_diff) / (self.config["sim_step"] * self.config["sim_steps_per_iter"]), -2, 2)

        # Check if the agent has reached a target
        target_dist = np.sqrt((torso_pos[0] - self.target[0]) ** 2 + (torso_pos[1] - self.target[1]) ** 2)
        velocity_rew = np.minimum(
            (self.prev_target_dist - target_dist) / (self.config["sim_step"] * self.config["sim_steps_per_iter"]),
            self.config["target_vel"]) / self.config["target_vel"]

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

        r_neg = {"inclination": np.sqrt(np.square(pitch) + np.square(roll)) * self.config["inclination_pen"],
                 "bobbing": np.square(zd) * 0.07 + np.square(thd) * 0.01 + np.square(phid) * 0.01,
                 "yaw_pen": np.square(tar_angle - yaw) * 0.10}

        r_pos = {"velocity_rew": np.clip(velocity_rew / (1 + abs(signed_deviation) * 3), -2, 2),
                 "height_rew": np.clip(torso_pos[2], 0, 0.00)}

        # print(r_pos["velocity_rew"])
        # r_pos = {"velocity_rew": np.clip(velocity_rew, -2, 2), "height_rew": np.clip(torso_pos[2], 0, 0.00)}

        r_pos_sum = sum(r_pos.values())
        r_neg_sum = np.maximum(np.minimum(sum(r_neg.values()) * (self.step_ctr > 5), r_pos_sum), 0)

        r = np.clip(r_pos_sum - r_neg_sum, -3, 3)


        if abs(r_pos_sum) > 3 or abs(r_neg_sum) > 3:
            print("!!WARNING!! REWARD IS ABOVE |3|, at step: {}  rpos = {}, rneg = {}".format(self.step_ctr, r_pos,
                                                                                              r_neg))

        # Assemble agent observation
        time_feature = [(float(self.step_ctr) / self.config["max_steps"]) * 2 - 1]
        compiled_obs = torso_quat, torso_vel, [signed_deviation], joint_angles, contacts, time_feature
        compiled_obs_flat = [item for sublist in compiled_obs for item in sublist]

        env_obs = np.array(compiled_obs_flat).astype(np.float32)

        done = self.step_ctr > self.config["max_steps"] or reached_target

        if np.abs(roll) > 1.57 or np.abs(pitch) > 1.57:
            # print("WARNING!! Absolute roll and pitch values exceed bounds: roll: {}, pitch: {}".format(roll, pitch))
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
            self.generate_rnd_env()
        else:
            self.terrain = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)
        self.create_targets()

        #sprint(p.getNumJoints(self.robot))

        # if self.config["randomize_env"]:
        #    self.robot = self.load_robot()

        # Reset episodal vars
        self.step_ctr = 0
        self.angle = 0
        self.episode_ctr += 1

        # self.config["target_spawn_mu"][0] = np.maximum(0., self.config["target_spawn_mu"][0] - 0.00005)
        # self.config["target_spawn_sigma"][0] = np.minimum(4., self.config["target_spawn_sigma"][0] + 0.00005)

        # Get heightmap height at robot position
        if self.terrain is None or self.config["terrain_name"] == "flat":
            spawn_height = 0
        else:
            spawn_height = 0.5 * np.max(
                self.terrain_hm[self.config["env_length"] // 2 - 3:self.config["env_length"] // 2 + 3,
                self.config["env_width"] // 2 - 3: self.config["env_width"] // 2 + 3]) * self.config["mesh_scale_vert"]

        # Random initial rotation
        rnd_rot = 0  # np.random.rand() * 0.3 - 0.15
        rnd_quat = p.getQuaternionFromAxisAngle([0, 0, 1], rnd_rot)

        [p.resetJointState(self.robot, i, 0, 0, physicsClientId=self.client_ID) for i in range(18)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, spawn_height + 0.3], rnd_quat,
                                          physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(18),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * 18,
                                    forces=[self.config["max_joint_force"]] * 18,
                                    physicsClientId=self.client_ID)

        self.prev_target_dist = np.sqrt((0 - self.target[0]) ** 2 + (0 - self.target[1]) ** 2)
        tar_angle = np.arctan2(self.target[1] - 0, self.target[0] - 0)

        for i in range(10):
            p.stepSimulation(physicsClientId=self.client_ID)

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs

    def test_phases(self):
        np.set_printoptions(precision=3)
        self.reset()

        y_dist = 0.1
        z_offset = -0.1

        while True:
            self.step([0] * self.act_dim)

            p.stepSimulation()
            time.sleep(self.config["sim_step"])

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

    def my_ikt(self, target_positions, y_offset):
        rotation_angles = [np.pi / 4, np.pi / 4, 0, 0, -np.pi / 4, -np.pi / 4]
        joint_angles = []
        for i, tp in enumerate(target_positions):
            tp_rotated = self.rotate_eef_pos(tp, rotation_angles[i], y_offset)
            joint_angles.extend(self.single_leg_ikt(tp_rotated))
        return joint_angles

    def rotate_eef_pos(self, eef_xyz, angle, y_offset):
        return [eef_xyz[0] * np.cos(angle), eef_xyz[0] * np.sin(angle) + y_offset, eef_xyz[2]]

    def single_leg_ikt(self, eef_xyz):
        x,y,z = eef_xyz

        assert -0.15 < x < 0.15
        assert 0.05 < y < 0.3
        assert -0.2 < z < 0.2

        q1 = 0.2137
        q2 = 0.785

        C = 0.052
        F = 0.0675
        T = 0.132

        psi = (np.arcsin(x/y))
        Cx = C * np.sin(psi)
        Cy = C * np.cos(psi)
        R = np.sqrt((x-Cx)**2 + (y-Cy)**2 + (z)**2)
        alpha = np.arcsin(-z/R)

        a = np.arccos((F**2 + R**2 - T**2) / (2 * F * R))
        b = np.arccos((F ** 2 + T ** 2 - R ** 2) / (2 * F * T))

        th1 = alpha - q1 - a
        th2 = np.pi - q2 - b

        return -psi, th1, th2

if __name__ == "__main__":
    import yaml

    with open("configs/eef.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env_config["w_1"] = True
    env_config["w_2"] = False
    env_config["phase_scalar"] = 1
    env = HexapodBulletEnv(env_config)
    env.test_phases()

    # while True:
    #     env.reset()