import yaml
with open("../../envs/bullet_nexabot/hexapod/configs/hexapod_config.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)