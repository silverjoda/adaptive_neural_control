import yaml
with open("configs/hexapod_config.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)