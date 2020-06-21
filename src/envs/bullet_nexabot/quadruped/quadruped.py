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
        if (animate):
          p.connect(p.GUI)
        else:
          p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if seed is not None:
            np.random.seed(seed)
            T.manual_seed(seed)

        self.animate = animate

        # Simulator parameters
        self.max_steps = max_steps
        self.obs_dim = 20
        self.act_dim = 12
        self.timeStep = 0.002

        # TODO: Keep making the bullet quadruped

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        self.robot = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "quadruped.urdf"))
        self.plane = p.loadURDF("plane.urdf")

        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)

    def get_obs(self):
        # Get cartpole observation
        obs = p.getJointState(self.robot, 0)
        return obs

    def render(self, close=False):
        pass

    def step(self, ctrl):
        p.setJointMotorControl2(self.robot, 0, p.POSITION_CONTROL)
        p.stepSimulation()

        self.step_ctr += 1

        # x, x_dot, theta, theta_dot
        obs = self.get_obs()
        x, y, z = obs[0:3]
        quat = obs[3:7]
        joints = obs[7:]

        ctrl_pen = np.square(ctrl[0]) * 0.001
        r = 0

        done = self.step_ctr > self.max_steps

        return np.array(obs), r, done, {}


    def reset(self):
        self.step_ctr = 0
        p.resetJointState(self.robot, 0, targetValue=0, targetVelocity=0)
        p.setJointMotorControl2(self.robot, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.robot, 1, p.VELOCITY_CONTROL, force=0)
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
                    time.sleep(0.01)
            print("Total episode reward: {}".format(cr))
        print("Total reward: {}".format(total_rew))



    def demo(self):
        # p.changeDynamics(self.robot, linkIndex=-1, lateralFriction=1)
        # p.changeDynamics(self.robot, linkIndex=3, lateralFriction=1)
        p.resetBasePositionAndOrientation(self.robot, [0, 0, .20], [0, 0, 0, 1])

        for i in range(100):
            for j in range(120):
                p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                            jointIndices=list(range(12)),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=[1] * 12,
                                            forces=[1] * 12)

                p.stepSimulation()
                time.sleep(0.002)


    def visualize_XML(self):
        #p.resetJointState(self.robot, 0, targetValue=0, targetVelocity=0)
        #p.setJointMotorControl2(self.robot, 0, p.VELOCITY_CONTROL, force=0)
        #p.setJointMotorControl2(self.robot, 1, p.VELOCITY_CONTROL, force=0)
        # TODO: TEST JOINTS ON 1 leg, then copy leg 4 times
        p.changeDynamics(self.robot, linkIndex=-1, lateralFriction=1)
        p.changeDynamics(self.robot, linkIndex=3, lateralFriction=1)
        p.resetBasePositionAndOrientation(self.robot, [0, 0, .15], [0, 0, 0, 1])
        while True:
            p.setJointMotorControlArray(bodyUniqueId=0,
                                        jointIndices=[0,1,2],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0, 1, 0],
                                        forces=[1,1,1])

            p.stepSimulation()
            time.sleep(0.002)

    def kill(self):
        p.disconnect()

    def close(self):
        self.kill()


if __name__ == "__main__":
    env = QuadrupedBulletEnv(animate=True)
    env.demo()