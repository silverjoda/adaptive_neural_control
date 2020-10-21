import fnmatch
NO_LIMITS = True
joint_limits = {"coxa" : [-0.4, 0.4],"femur" : [-1.6, -0.6],"tibia" : [0.9, 1.9]}
if NO_LIMITS:
    joint_limits = {"coxa": [-5., 5.], "femur": [-5., 5.], "tibia": [-5., 5.]}
input_filename = "hexapod.urdf"
output_filename = "hexapod_wide_range.urdf"

with open(input_filename, "r") as in_file: #
    buf = in_file.readlines()

with open(output_filename, "w") as out_file:
    for line in buf:
        if line.rstrip('\n').endswith('<!--coxa-->'):
            out_file.write('      <limit effort="1.4" lower="{}" upper="{}" velocity="1"/><!--coxa-->\n'.format(joint_limits["coxa"][0],joint_limits["coxa"][1]))
        elif line.rstrip('\n').endswith('<!--femur-->'):
            out_file.write('      <limit effort="1.4" lower="{}" upper="{}" velocity="1"/><!--femur-->\n'.format(joint_limits["femur"][0],joint_limits["femur"][1]))
        elif line.rstrip('\n').endswith('<!--tibia-->'):
            out_file.write('      <limit effort="1.4" lower="{}" upper="{}" velocity="1"/><!--tibia-->\n'.format(joint_limits["tibia"][0],joint_limits["tibia"][1]))
        else:
            out_file.write(line)