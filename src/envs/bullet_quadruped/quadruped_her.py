import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import torch as T
import gym
from gym import spaces, error
from gym.utils import closer
from stable_baselines import HER, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv

class QuadrupedBulletEnv(gym.GoalEnv):
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

        if self.config["animate"]:
            self.client_ID = p.connect(p.GUI)
            print(" --Starting GUI mode-- ")
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        self.obs_dim = 23
        self.act_dim = 12

        self.observation_space = spaces.Dict({'observation' : spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32),
                                             'achieved_goal' : spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                                             'desired_goal' : spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)})
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

        self.joints_rads_low = np.array(config["joints_rads_low"] * 4)
        self.joints_rads_high = np.array(config["joints_rads_high"] * 4)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        self.robot = None
        self.robot = self.load_robot()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client_ID)

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

    def set_randomize_env(self, rnd):
        self.config["randomize_env"] = rnd

    def load_robot(self):
        # Remove old robot
        if self.robot is not None:
            p.removeBody(self.robot)

        # Randomize robot params
        self.randomized_robot_params = {"mass": 1 + (np.random.rand() * 1.0 - 0.5) * self.config["randomize_env"],
                             "tibia_fl": 0.12 + (np.random.rand() * 0.12 - 0.06) * self.config["randomize_env"],
                             "tibia_fr": 0.12 + (np.random.rand() * 0.12 - 0.06) * self.config["randomize_env"],
                             "tibia_rl": 0.12 + (np.random.rand() * 0.12 - 0.06) * self.config["randomize_env"],
                             "tibia_rr": 0.12 + (np.random.rand() * 0.12 - 0.06) * self.config["randomize_env"],
                             "max_joint_force": 1.4 + np.random.rand() * 1. * self.config["randomize_env"]}

        # Write params to URDF file
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]), "r") as in_file:
            buf = in_file.readlines()

        index = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"]).find('.urdf')
        output_urdf = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"])[:index] + '_rnd' + os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config["urdf_name"])[index:]

        # Change link lengths in urdf
        with open(output_urdf, "w") as out_file:
            for line in buf:
                if line.rstrip('\n').endswith('<!--tibia_fl-->'):
                    out_file.write(f'          <cylinder radius="0.01" length="{self.randomized_robot_params["tibia_fl"]}"/>\n')
                elif line.rstrip('\n').endswith('<!--tibia_fr-->'):
                    out_file.write(f'          <cylinder radius="0.01" length="{self.randomized_robot_params["tibia_fr"]}"/>\n')
                elif line.rstrip('\n').endswith('<!--tibia_rl-->'):
                    out_file.write(f'          <cylinder radius="0.01" length="{self.randomized_robot_params["tibia_rl"]}"/>\n')
                elif line.rstrip('\n').endswith('<!--tibia_rr-->'):
                    out_file.write(f'          <cylinder radius="0.01e" length="{self.randomized_robot_params["tibia_rr"]}"/>\n')

                elif line.rstrip('\n').endswith('<!--tibia_fl_2-->'):
                    out_file.write(f'        <origin xyz=".0 {self.randomized_robot_params["tibia_fl"] / 2.} 0.0" rpy="1.5708 0 0" />"/>\n')
                elif line.rstrip('\n').endswith('<!--tibia_fr_2-->'):
                    out_file.write(f'        <origin xyz=".0 {-self.randomized_robot_params["tibia_fr"] / 2.} 0.0" rpy="1.5708 0 0" />"/>\n')
                elif line.rstrip('\n').endswith('<!--tibia_rl_2-->'):
                    out_file.write(f'        <origin xyz=".0 {self.randomized_robot_params["tibia_rl"] / 2.} 0.0" rpy="1.5708 0 0" />"/>\n')
                elif line.rstrip('\n').endswith('<!--tibia_rr_2-->'):
                    out_file.write(f'        <origin xyz=".0 {-self.randomized_robot_params["tibia_rr"] / 2.} 0.0" rpy="1.5708 0 0" />"/>\n')
                else:
                    out_file.write(line)

        # Load urdf
        robot = p.loadURDF(output_urdf, physicsClientId=self.client_ID)

        if self.config["randomize_env"]:
            # Change base mass
            p.changeDynamics(robot, -1, mass=self.randomized_robot_params["mass"])

        for i in range(4):
            p.changeDynamics(robot, 3 * i + 2, lateralFriction=self.config["lateral_friction"])
        p.changeDynamics(robot, -1, lateralFriction=self.config["lateral_friction"])

        return robot

    def render(self, close=False):
        pass

    def step(self, ctrl):
        ctrl_scaled = self.norm_to_rads(np.tanh(ctrl))
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(12),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=ctrl_scaled,
                                    forces=[self.config["max_joint_force"]] * 12,
                                    positionGains=[0.02] * 12,
                                    velocityGains=[0.01] * 12,
                                    physicsClientId=self.client_ID)

        for _ in range(self.config["sim_steps_per_iter"]):
            p.stepSimulation(physicsClientId=self.client_ID)
            if self.config["animate"]: time.sleep(0.004)

        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts = self.get_obs()
        xd, yd, zd = torso_vel
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        velocity_rew = np.minimum(xd, self.config["target_vel"]) / self.config["target_vel"]

        r = -(np.square(yaw) * 0.0 + \
                np.square(roll) * 0.3 + \
                np.square(pitch) * 0.3 + \
                np.square(zd) * 0.3)

        env_obs = {'observation' : np.concatenate((self.rads_to_norm(joint_angles), torso_quat, torso_pos, contacts)).astype(np.float32),
                   'achieved_goal' : np.array(torso_pos[:2]).astype(np.float32),
                   'desired_goal' : np.array([2.0,0]).astype(np.float32)}

        self.step_ctr += 1
        overturned = np.abs(roll) > 1.57 or np.abs(pitch) > 1.57
        done = self.step_ctr > self.config["max_steps"] or overturned

        return env_obs, r, done, {}

    def reset(self):
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))


        if self.config["randomize_env"]:
            self.robot = self.load_robot()

        self.step_ctr = 0

        joint_init_pos_list = self.norm_to_rads([0] * 12)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(12)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, .15], [0, 0, 0, 1], physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(12),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * 12,
                                    forces=[1.4] * 12,
                                    physicsClientId=self.client_ID)
        for i in range(10):
            p.stepSimulation(physicsClientId=self.client_ID)

        obs, _, _, _ = self.step(np.zeros(self.act_dim))
        return obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.sqrt(np.square(achieved_goal - desired_goal).sum())

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
        n_steps = 30
        VERBOSE = True
        while True:
            t1 = time.time()
            sc = 1.0
            test_acts = [[0, 0, 0], [0, sc, sc], [0, -sc, -sc], [0, sc, -sc], [0, -sc, sc], [sc, 0, 0], [-sc, 0, 0]]
            for i, a in enumerate(test_acts):
                for j in range(n_steps):
                    # a = list(np.random.randn(3))
                    scaled_obs, _, _, _ = self.step(np.array(a * 4))
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
    with open("configs/default.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env_config["animate"] = False

    model_class = TD3  # works also with SAC, DDPG and TD3

    env = QuadrupedBulletEnv(env_config)

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

    # Wrap the model
    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                verbose=1)
    # Train the model
    model.learn(200000)

    model.save("./her_quadruped_env")
    env.close()
    print("FINISHED TRAININIG")

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    env_config["animate"] = True
    env = QuadrupedBulletEnv(env_config)
    model = HER.load('./her_quadruped_env', env=env)

    print("STARTING EVALUATION")
    obs = env.reset()
    for i in range(100):
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)

            if done:
                obs = env.reset()
                break

    #env.test_leg_coordination()