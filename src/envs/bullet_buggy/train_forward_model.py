import os
import glob
import yaml
import numpy as np
import torch as T
import torch.nn as nn
import torch.functional as F

class ForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.l1 = nn.Linear(self.config["input_dim"], self.config["hidden_dim"])
        self.l2 = nn.Linear(self.config["hidden_dim"], self.config["hidden_dim"])
        self.l3 = nn.Linear(self.config["hidden_dim"], self.config["output_dim"])

        self.non_linearity = eval(self.config["non_linearity"])

    def forward(self, x):
        feat1 = self.non_linearity(self.l1(x))
        feat2 = self.non_linearity(self.l2(feat1))
        out = self.l1(feat2)
        return out

    def predict(self, obs, act):
        x = T.tensor(np.concatenate((obs, act)), dtype=T.float32).unsqueeze(0)
        return self.forward(x)

    def predict_batch(self, obs, act):
        x = T.tensor(np.concatenate((obs, act), axis=1), dtype=T.float32)
        return self.forward(x)

class ForwardModel:
    def __init__(self, config):
        self.config = config

        self.load_data()
        self.make_train_val_data()

        self.NN = ForwardModel(config)

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = T.optim.Adam(self.NN.parameters(),
                                 lr=self.config["learning_rate"],
                                 weight_decay=self.config["weight_decay"])

    def load_data(self):
        data_types_list = ["action", "angular_vel", "position", "rotation", "timestamp", "vel",]
        for dt in data_types_list:
            data_list = []
            for name in glob.glob(f'data/train/*{dt}.npy'):
                data_list.append(np.load(name))
            vars(self)[dt + "_data"] = np.concatenate(data_list, axis=1)

    def make_train_val_data(self):
        obs, labels = self._preprocess_data()

        n_train = int(len(obs) * self.config['trn_val_ratio'])
        trn_indeces = np.random.choice(range(len(obs)), n_train, replace=False)
        trn_mask = np.zeros(n_train, dtype=np.bool)
        trn_mask[trn_indeces] = True
        val_indeces = np.arange(len(obs))[~trn_mask]

        self.obs_trn = obs[trn_indeces]
        self.labels_trn = labels[trn_indeces]
        self.obs_val = obs[val_indeces]
        self.labels_val = labels[val_indeces]

    def _preprocess_data(self):
        # Calculate deltas
        rotation_delta = self.rotation_data[2:, 2:3] - self.rotation_data[1:-1, 2:3]
        vel_delta = self.vel_data[2:, 2:3] - self.vel_data[1:-1, 2:3]
        angular_vel_delta = self.angular_vel_data[2:, 2:3] - self.angular_vel_data[1:-1, 2:3]

        obs = np.concatenate((self.vel_data[1:-1, 0:2],
                              self.angular_vel_data[1:-1, 2:3],
                              self.action_data[:-2, 2:3]))
        labels = np.concatenate((rotation_delta[2:, 2:3],
                                 vel_delta[2:, 0:2],
                                 angular_vel_delta[2:, 2:3]))

        return obs, labels

    def get_batch(self, batchsize):
        trn_indeces = np.random.choice(range(len(self.obs_trn)), batchsize, replace=False)
        obs_batch = self.obs_trn[trn_indeces]
        labels_batch = self.labels_trn[trn_indeces]
        return obs_batch, labels_batch

    def train(self):
        for t in range(self.config["n_trn_iters"]):
            x_trn, y_trn = self.get_batch(self.config["trn_batchsize"])

            y_pred = self.NN.predict_batch(x_trn)
            loss = self.criterion(y_pred, y_trn)

            if t % 100 == 99:
                print(t, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Done training")

    def eval(self):
        with T.no_grad():
            y_pred = self.NN.predict_batch(self.obs_val)
            loss = self.criterion(y_pred, self.labels_val)
        print(f"Evaluation: loss = {loss.item()}")

    def save_model(self):
        if not os.path.exists("models"):
            os.mkdir("models/")
        T.save(self.NN.state_dict(), "models/")

    def load_model(self, filename):
        try:
            self.NN.load_state_dict(T.load(f"models/{filename}"))
        except:
            print("Failed to load trained model")

if __name__=="__main__":
    with open("configs/model_training.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    fm = ForwardModel(config)
    fm.load_data()