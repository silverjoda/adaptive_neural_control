import numpy as np
np.set_printoptions(precision=5)

joints_rads_low = np.array([-0.4, -1.6, 0.9] )
joints_rads_high = np.array([0.4, -0.6, 1.9])
joints_rads_diff = joints_rads_high - joints_rads_low

joints = [0.4, -0.6, 0.4]

sjoints = np.array(joints)
sjoints = ((sjoints - joints_rads_low) / joints_rads_diff) * 2 - 1

print(joints)
print(sjoints)