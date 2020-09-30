import yaml
with open("../../envs/bullet_hexapod/configs/hexapod_config.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)