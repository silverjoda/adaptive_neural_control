import math
import os
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data

from gym import spaces
from opensimplex import OpenSimplex
import matplotlib.pyplot as plt

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
        self.act_dim = self.config["act_dim"]
        self.obs_dim = 1

        self.x_mult = self.config["x_mult"]
        self.y_offset = self.config["y_offset"]
        self.z_mult = self.config["z_mult"]
        self.z_offset = self.config["z_offset"]
        self.z_lb = self.config["z_lb"]

        self.dyn_z_lb_array = np.array([float(self.z_lb)] * 6)
        self.poc_array = np.array([float(self.z_lb)] * 6)

        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-3, high=3, shape=(self.act_dim,), dtype=np.float32)

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.urdf_name = config["urdf_name"]
        self.robot = self.load_robot()

        if config["terrain_name"] == "flat":
            self.terrain = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)
        else:
            self.terrain = self.generate_rnd_env()

        self.step_ctr = 0
        self.angle = 0
        self.create_targets()

    def set_seed(self, np_seed, T_seed):
        np.random.seed(np_seed)

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
            hm = np.random.randint(0, self.config["tiles_height"], # 15
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

    def render(self, close=False, mode=None):
        if self.config["animate"]:
            time.sleep(self.config["sim_step"])

    def load_robot(self):
        # Remove old robot
        if not hasattr(self, 'robot'):
            self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.urdf_name), physicsClientId=self.client_ID)

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
            p.changeDynamics(self.robot, 3 * i + 2, lateralFriction=self.randomized_params["lateral_friction"], physicsClientId=self.client_ID)

        return self.robot

    def step(self, ctrl_raw, render=False):
        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts, ctct_torso = self.get_obs()
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)
        tar_angle = np.arctan2(self.target[1] - torso_pos[1], self.target[0] - torso_pos[0])
        signed_deviation = np.clip(yaw - tar_angle, -0.4, 0.4)

        phases = ctrl_raw[0:6]

        x_mult_arr = [self.x_mult + signed_deviation * self.config["turn_coeff"], self.x_mult - signed_deviation * self.config["turn_coeff"]] * 3

        targets = []
        for i in range(6):
            x_cyc = np.sin(self.angle * 2 * np.pi + phases[i])
            z_cyc = np.cos(self.angle * 2 * np.pi + phases[i])

            target_x = x_cyc * x_mult_arr[i]
            target_y = self.y_offset

            # if i==3 and z_cyc < 0:
            #     contacts[i] = 1

            if x_cyc < 0 and z_cyc > 0.0:
                self.dyn_z_lb_array[i] = self.z_lb

            if contacts[i] < 0:
                self.dyn_z_lb_array[i] = z_cyc
                self.poc_array[i] = 1
            else:
                if self.poc_array[i] == 1:
                    self.poc_array[i] = z_cyc
                self.dyn_z_lb_array[i] = self.poc_array[i] - self.config["z_pressure_coeff"]

            target_z = np.maximum(z_cyc, self.dyn_z_lb_array[i]) * self.z_mult + self.z_offset
            #target_z = z_cyc * self.z_mult + self.z_offset
            #x_pitch_rot = target_x * np.cos(-pitch * self.config["pitch_rot_coeff"]) - target_z * np.sin(-pitch * self.config["pitch_rot_coeff"])
            #z_pitch_rot = target_x * np.sin(-pitch * self.config["pitch_rot_coeff"]) + target_z * np.cos(-pitch * self.config["pitch_rot_coeff"])

            targets.append([target_x, target_y, target_z])

        #rotation_overlay = np.clip(np.array(ctrl_raw[6:12]), -np.pi, np.pi)
        joint_angles = self.my_ikt_robust(targets)
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
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        # Orientation angle
        tar_angle = np.arctan2(self.target[1] - torso_pos[1], self.target[0] - torso_pos[0])
        yaw_deviation = np.min((abs((yaw % 6.283) - (tar_angle % 6.283)), abs(yaw - tar_angle)))

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
        else:
            reached_target = False
            self.prev_target_dist = target_dist

        # if self.config['locomotion_mode'] == "straight":
        #     r = velocity_rew - abs(zd) * 0.2 - abs(roll) * 0.3 - abs(pitch) * 0.3
        # if self.config['locomotion_mode'] == "ccw":
        #     r = np.minimum(psid, 0.8) - abs(roll) * 0.2 - abs(pitch) * 0.2 - abs(zd) * 0.2 - abs(phid) * 0.03 - abs(
        #         thd) * 0.03
        # if self.config['locomotion_mode'] == "cw":
        #     r = np.maximum(-psid, -0.8) - abs(roll) * 0.2 - abs(pitch) * 0.2 - abs(zd) * 0.2 - abs(phid) * 0.03 - abs(
        #         thd) * 0.03

        r = velocity_rew
        r_neg = {"inclination": np.sqrt(np.square(pitch) + np.square(roll)) * 0.1,  # 0.1
                 "bobbing": np.sqrt(np.square(zd)) * 0.1,  # 0.2
                 "yaw_pen": np.square(tar_angle - yaw) * 0.1}
        r_pos = {"task_rew": r,
                 "height_rew": np.clip(torso_pos[2], 0, 0.00)}

        r_pos_sum = sum(r_pos.values())
        r_neg_sum = np.maximum(np.minimum(sum(r_neg.values()) * (self.step_ctr > 5), r_pos_sum), 0)
        r = r_pos_sum - r_neg_sum

        # Calculate relative positions of targets
        relative_target = self.target[0] - torso_pos[0], self.target[1] - torso_pos[1]

        # Orientation angle
        tar_angle = np.arctan2(self.target[1] - torso_pos[1], self.target[0] - torso_pos[0])
        yaw_deviation = np.min((abs((yaw % 6.283) - (tar_angle % 6.283)), abs(yaw - tar_angle)))
        signed_deviation = yaw - tar_angle

        # Assemble agent observation
        env_obs = np.array([signed_deviation]).astype(np.float32)
        done = self.step_ctr > self.config["max_steps"] or reached_target

        if np.abs(roll) > 1.57 or np.abs(pitch) > 1.57:
            print("WARNING!! Absolute roll and pitch values exceed bounds: roll: {}, pitch: {}".format(roll, pitch))
            done = True

        if abs(torso_pos[0]) > 6 or abs(torso_pos[1]) > 6 or abs(torso_pos[2]) > 1.5:
            print("WARNING: TORSO OUT OF RANGE!!")
            done = True

        return env_obs, r, done, {}

    def reset(self, force_randomize=None):
        if self.config["randomize_env"]:
            self.robot = self.load_robot()

        # Reset episodal vars
        self.step_ctr = 0
        self.angle = 0

        # Change heightmap with small probability
        if np.random.rand() < self.config["env_change_prob"] and not self.config["terrain_name"] == "flat":
            self.terrain = self.generate_rnd_env()

        # Get heightmap height at robot position
        if self.terrain is None or self.config["terrain_name"] == "flat":
            spawn_height = 0
        else:
            spawn_height = 0.5 * np.max(self.terrain_hm[self.config["env_length"] // 2 - 3:self.config["env_length"] // 2 + 3, self.config["env_width"] // 2 - 3 : self.config["env_width"] // 2 + 3]) * self.config["mesh_scale_vert"]

        # Random initial rotation
        rnd_rot = 0 #np.random.rand() * 0.3 - 0.15
        rnd_quat = p.getQuaternionFromAxisAngle([0, 0, 1], rnd_rot)

        [p.resetJointState(self.robot, i, 0, 0, physicsClientId=self.client_ID) for i in range(18)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, spawn_height + 0.3], rnd_quat, physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(18),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * 18,
                                    forces=[self.config["max_joint_force"]] * 18,
                                    physicsClientId=self.client_ID)

        self.update_targets()
        self.prev_target_dist = np.sqrt((0 - self.target[0]) ** 2 + (0 - self.target[1]) ** 2)
        tar_angle = np.arctan2(self.target[1] - 0, self.target[0] - 0)

        for i in range(10):
            p.stepSimulation(physicsClientId=self.client_ID)

        return  np.zeros(self.obs_dim)

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

    def test_my_ikt(self):
        np.set_printoptions(precision=3)
        self.reset()

        leg_pts = []
        step_ctr = 0
        while True:
            self.step([0]*self.act_dim)

            _,_,torso_vel,_, joint_angles, _,_,_,_ = self.get_obs()

            if step_ctr > 10:
                leg_pts.append(self.single_leg_dkt(joint_angles[9:12]))

            if step_ctr == 60:
                break
            step_ctr += 1

        x = [leg_pt[0] for leg_pt in leg_pts]
        z = [leg_pt[2] for leg_pt in leg_pts]
        colors = np.random.rand(len(leg_pts))
        plt.scatter(x, z, c=colors, alpha=0.5)
        plt.show()

    def close(self):
        p.disconnect(physicsClientId=self.client_ID)

    def my_ikt(self, target_positions, rotation_overlay=None):
        #raise NotImplementedError
        rotation_angles = np.array([np.pi/4,np.pi/4,0,0,-np.pi/4,-np.pi/4])
        if rotation_overlay is not None:
            rotation_angles += rotation_overlay
        joint_angles = []
        for i, tp in enumerate(target_positions):
            tp_rotated = self.rotate_eef_pos(tp, rotation_angles[i], tp[1])
            joint_angles.extend(self.single_leg_ikt(tp_rotated))
        return joint_angles

    def my_ikt_robust(self, target_positions, rotation_overlay=None):
        #raise NotImplementedError
        def find_nearest_valid_point(xyz_query, rot_angle=0):
            sol = self.single_leg_ikt(xyz_query)
            if not np.isnan(sol).any(): return sol

            cur_valid_sol = None
            cur_xyz_query = xyz_query
            cur_delta = 0.03
            n_iters = 10

            if xyz_query[2] > -0.1:
                search_dir = 1
            else:
                search_dir = -1

            cur_xyz_query[0] = cur_xyz_query[0] - cur_delta * search_dir * np.sin(rot_angle)
            cur_xyz_query[1] = cur_xyz_query[1] + cur_delta * search_dir * np.cos(rot_angle)
            for _ in range(n_iters):
                sol = self.single_leg_ikt(cur_xyz_query)
                if not np.isnan(sol).any(): # If solution is good
                    cur_valid_sol = sol
                    cur_delta /= 2
                    cur_xyz_query[0] = cur_xyz_query[0] + cur_delta * search_dir * np.sin(rot_angle)
                    cur_xyz_query[1] = cur_xyz_query[1] - cur_delta * search_dir * np.cos(rot_angle)
                else:
                    if cur_valid_sol is not None:
                        cur_delta /= 2
                    cur_xyz_query[0] = cur_xyz_query[0] - cur_delta * search_dir * np.sin(rot_angle)
                    cur_xyz_query[1] = cur_xyz_query[1] + cur_delta * search_dir * np.cos(rot_angle)

            assert cur_valid_sol is not None and not np.isnan(cur_valid_sol).any()
            return cur_valid_sol

        rotation_angles = np.array([np.pi/4,np.pi/4,0,0,-np.pi/4,-np.pi/4])
        if rotation_overlay is not None:
            rotation_angles += rotation_overlay
        joint_angles = []
        for i, tp in enumerate(target_positions):
            tp_rotated = self.rotate_eef_pos(tp, rotation_angles[i], tp[1])
            joint_angles.extend(find_nearest_valid_point(tp_rotated, rotation_angles[i]))
        return joint_angles

    def rotate_eef_pos(self, eef_xyz, angle, y_offset):
        return [eef_xyz[0] * np.cos(angle), eef_xyz[0] * np.sin(angle) + y_offset, eef_xyz[2]]

    def single_leg_ikt(self, eef_xyz):
        x,y,z = eef_xyz

        q1 = 0.2137
        q2 = 0.785

        C = 0.052
        F = 0.0675
        T = 0.132

        psi = np.arctan(x/y)
        Cx = C * np.sin(psi)
        Cy = C * np.cos(psi)
        R = np.sqrt((x-Cx)**2 + (y-Cy)**2 + (z)**2)
        alpha = np.arcsin(-z/R)

        a = np.arccos((F**2 + R**2 - T**2) / (2 * F * R))
        b = np.arccos((F ** 2 + T ** 2 - R ** 2) / (2 * F * T))

        #if np.isnan(a) or np.isnan(b):
        #    print(a,b)

        assert 0 < a < np.pi or np.isnan(a)
        assert 0 < b < np.pi or np.isnan(b)

        th1 = alpha - q1 - a
        th2 = np.pi - q2 - b

        assert th2 + q2 > 0 or np.isnan(th2)

        return -psi, th1, th2

    def single_leg_dkt(self, angles):
        psi, th1, th2 = angles

        q1 = 0.2137
        q2 = 0.785

        C = 0.052
        F = 0.0675
        T = 0.132

        Ey_flat = (C + F * np.cos(q1 + th1) + T * np.cos(q1 + th1 + q2 + th2))

        Ez = - F * np.sin(q1 + th1) - T * np.sin(q1 + th1 + q2 + th2)
        Ey = Ey_flat * np.cos(psi)
        Ex = Ey_flat * np.sin(-psi)

        return (Ex, Ey, Ez)

    def test_kinematics(self):
        #np.set_printoptions(precision=5)

        # psi_range = np.linspace(-0.6, 0.6, 10)
        # th1_range = np.linspace(-np.pi / 2, np.pi / 2, 10)
        # th2_range = np.linspace(-np.pi / 2, np.pi / 2, 10)
        #
        # # Do a sweep first and see minimum and maximum
        # ex_list = []
        # ey_list = []
        # ez_list = []
        # for psi in psi_range:
        #     for th1 in th1_range:
        #         for th2 in th2_range:
        #             ex, ey, ez = self.single_leg_dkt((psi, th1, th2))
        #             ex_list.append(ex)
        #             ey_list.append(ey)
        #             ez_list.append(ez)
        # print(f"Ex min: {min(ex_list)}, max: {max(ex_list)}  Ey min: {min(ey_list)}, max: {max(ey_list)}   Ez min: {min(ez_list)}, max: {max(ez_list)} ")
        # exit()

        ex_range = np.linspace(-0.1, 0.1, 20)
        #ey_range = [0.1]
        ey_range = np.linspace(0.05, 0.25, 20)
        ez_range = np.linspace(-0.25, 0.1, 40)

        valid_eef_pts = []
        # Compare DKT and IKT
        for ex in ex_range:
            for ey in ey_range:
                for ez in ez_range:
                    psi, th1, th2 = self.single_leg_ikt((ex, ey, ez))
                    ex_ikt, ey_ikt, ez_ikt = self.single_leg_dkt((psi, th1, th2))

                    print(f"For eef pose: {format(ex, '.4f')}, {format(ey, '.4f')}, {format(ez, '.4f')},"
                              f" ikt gave: {format(psi, '.4f')}, {format(th1, '.4f')}, {format(th2, '.4f')}",
                              f" for ikt angle dkt gave: {format(ex_ikt, '.4f')}, {format(ey_ikt, '.4f')}, {format(ez_ikt, '.4f')}")
                    if np.isclose(np.array((ex, ey, ez)), np.array((ex_ikt, ey_ikt, ez_ikt)), atol=0.001).all() \
                            and not np.isnan((ex_ikt, ey_ikt, ez_ikt)).any():
                        valid_eef_pts.append([ex,ey,ez])


        print(f"Valid pts: {len(valid_eef_pts)}")
        x, y, z = list(zip(*valid_eef_pts))

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x,y,z, marker='*')
        ax.set_title('Scatter plot')
        plt.show()

    def demo_kinematics(self):
        pass

if __name__ == "__main__":
    import yaml
    with open("configs/eef.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = True
    env = HexapodBulletEnv(env_config)
    #env.test_my_ikt()
    #env.test_kinematics()
    env.test_my_ikt()
