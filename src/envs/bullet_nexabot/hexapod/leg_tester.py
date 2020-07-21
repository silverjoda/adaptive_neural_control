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

import torch as T
import torch.nn as nn
import torch.nn.functional as F

class LegModelLSTM(nn.Module):
    def __init__(self, n_inputs=3, n_actions=3):
        nn.Module.__init__(self)
        n_hidden = 12
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.lstm = nn.LSTM(n_hidden, n_hidden, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(n_hidden, n_actions)
        self.activ_fn = nn.ReLU()

    def forward(self, x):
        x = self.activ_fn(self.fc1(x))
        x, h = self.lstm(x, None)
        x = self.fc2(x)
        return x, h

def scale_joints(joints):
    sjoints = np.array(joints)
    sjoints = ((sjoints - joints_rads_low) / joints_rads_diff) * 2 - 1
    return sjoints

def scale_action(action):
    return (np.array(action) * 0.5 + 0.5) * joints_rads_diff + joints_rads_low

if __name__ == "__main__":
    animate = False
    if (animate):
        client_ID = p.connect(p.GUI)
        print(" --Starting GUI mode-- ")
    else:
        client_ID = p.connect(p.DIRECT)
    assert client_ID != -1, "Physics client failed to connect"

    # Normal
    joints_rads_low = np.array([-0.4, -1.6, 0.9])
    joints_rads_high = np.array([0.4, -0.6, 1.9])
    joints_rads_diff = joints_rads_high - joints_rads_low

    p.setGravity(0, 0, -9.8, physicsClientId=client_ID)
    p.setRealTimeSimulation(0, physicsClientId=client_ID)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_ID)

    leg_urdf_name = "hexapod_leg.urdf"
    leg = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), leg_urdf_name),
                       physicsClientId=client_ID)

    leg_model_nn = LegModelLSTM(n_inputs=6, n_actions=3)
    optim = T.optim.Adam(leg_model_nn.parameters())
    batchsize = 30
    n_iters = 1000
    episode_len = 100
    max_joint_force = 1.3
    sim_steps_per_iter = 24

    joints = []
    acts = []
    for iter in range(n_iters):
        ep_joints = []
        ep_acts = []
        # Sample batchsize episodes
        [p.resetJointState(leg, i, np.random.randn(), 0, physicsClientId=client_ID) for i in range(3)]
        for st in range(episode_len):
            # Joint
            obs = p.getJointStates(leg, range(3), physicsClientId=client_ID)  # pos, vel, reaction(6), prev_torque
            joint_angles = []
            joint_velocities = []
            joint_torques = []
            for o in obs:
                joint_angles.append(o[0])
                joint_velocities.append(o[1])
                joint_torques.append(o[3])
            joint_angles_normed = scale_joints(joint_angles)
            ep_joints.append(joint_angles_normed)

            # Action
            act_normed = np.clip(np.random.randn(3), -1, 1)
            scaled_action = scale_action(act_normed)
            ep_acts.append(act_normed)

            # Simulate
            for i in range(3):
                p.setJointMotorControl2(bodyUniqueId=leg,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=scaled_action[i],
                                        force=max_joint_force,
                                        positionGain=0.1,
                                        velocityGain=0.1,
                                        maxVelocity=2.0,
                                        physicsClientId=client_ID)

            for i in range(sim_steps_per_iter):
                p.stepSimulation(physicsClientId=client_ID)
                if (animate) and True: time.sleep(0.0038)

        joints.append(ep_joints)
        acts.append(ep_acts)

        if iter % batchsize == 0 and iter > 0:
            # Forward pass
            joints_T = T.tensor([j[:-1] for j in joints] , dtype=T.float32)
            joints_next_T = T.tensor([j[1:] for j in joints], dtype=T.float32)
            acts_T = T.tensor([j[:-1] for j in acts] , dtype=T.float32)
            pred, _ = leg_model_nn(T.cat((joints_T, acts_T), dim=2))

            joints = []
            acts = []

            # update
            optim.zero_grad()
            loss = F.mse_loss(pred, joints_next_T)
            optim.step()

            print("Iter: {}/{}, loss: {}".format(iter, n_iters, loss))
