import numpy as np
import pybullet as p

class PIDController:
    def __init__(self, config):
        self.config = config
        self.setup_stabilization_control()

    def setup_stabilization_control(self):
        self.e_roll_prev = 0
        self.e_pitch_prev = 0
        self.e_yaw_prev = 0

        self.e_roll_accum = 0
        self.e_pitch_accum = 0

    def calculate_stabilization_action(self, orientation, angular_velocities, targets):
        roll, pitch, _ = p.getEulerFromQuaternion(orientation)
        roll_vel, pitch_vel, yaw_vel = angular_velocities
        t_throttle, t_roll, t_pitch, t_yaw_vel = targets

        # Increase t_yaw_vel because it's slow as shit
        t_yaw_vel *= 5

        # Target errors
        e_roll = t_roll - roll
        e_pitch = t_pitch - pitch
        e_yaw = t_yaw_vel - yaw_vel

        self.e_roll_accum = self.e_roll_accum * self.config["i_decay"] + e_roll
        self.e_pitch_accum = self.e_pitch_accum * self.config["i_decay"] + e_pitch

        # Desired correction action
        roll_act = e_roll * self.config["p_roll"] + (e_roll - self.e_roll_prev) * self.config[
            "d_roll"]  # + self.e_roll_accum * self.config["i_roll"]
        pitch_act = e_pitch * self.config["p_pitch"] + (e_pitch - self.e_pitch_prev) * self.config[
            "d_pitch"]  # + self.e_pitch_accum * self.config["i_pitch"]
        yaw_act = e_yaw * self.config["p_yaw"] + (e_yaw - self.e_yaw_prev) * self.config["d_yaw"]

        self.e_roll_prev = e_roll
        self.e_pitch_prev = e_pitch
        self.e_yaw_prev = e_yaw

        m_1_act_total = + roll_act - pitch_act + yaw_act
        m_2_act_total = - roll_act - pitch_act - yaw_act
        m_3_act_total = + roll_act + pitch_act - yaw_act
        m_4_act_total = - roll_act + pitch_act + yaw_act

        # Translate desired correction actions to servo commands
        m_1 = np.clip(t_throttle + m_1_act_total, 0, 1)
        m_2 = np.clip(t_throttle + m_2_act_total, 0, 1)
        m_3 = np.clip(t_throttle + m_3_act_total, 0, 1)
        m_4 = np.clip(t_throttle + m_4_act_total, 0, 1)

        if np.max([m_1, m_2, m_3, m_4]) > 1.1:
            print("Warning: motor commands exceed 1.0. This signifies an error in the system", m_1, m_2, m_3, m_4,
                  t_throttle)

        return m_1, m_2, m_3, m_4
