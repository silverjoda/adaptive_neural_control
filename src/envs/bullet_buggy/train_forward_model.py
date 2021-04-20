import os
import glob
import yaml
import numpy as np
import torch as T
import torch.nn as nn
import torch.functional as F
import quaternion

class ForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.l1 = nn.Linear(self.config["input_dim"], self.config["hidden_dim"])
        self.l2 = nn.Linear(self.config["hidden_dim"], self.config["hidden_dim"])
        self.l3 = nn.Linear(self.config["hidden_dim"], self.config["output_dim"])

        self.non_linearity = eval(self.config["non_linearity"])()

    def forward(self, x):
        feat1 = self.non_linearity(self.l1(x))
        feat2 = self.non_linearity(self.l2(feat1))
        out = self.l3(feat2)
        return out

    def predict(self, obs):
        x = T.tensor(obs, dtype=T.float32).unsqueeze(0)
        return self.forward(x)

    def predict_batch(self, obs):
        x = T.tensor(obs, dtype=T.float32)
        return self.forward(x)

class ForwardModelTrainer:
    def __init__(self, config):
        self.config = config

        self.load_data()
        self.make_train_val_data()

        self.NN = ForwardNet(config)

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = T.optim.Adam(self.NN.parameters(),
                                 lr=self.config["learning_rate"],
                                 weight_decay=self.config["weight_decay"])

    def load_data(self):
        data_types_list = ["action", "angular", "position", "rotation", "timestamp", "vel"]
        for dt in data_types_list:
            data_list = []
            for name in glob.glob(f'data/train/*{dt}.npy'):
                if dt == "timestamp":
                    data_list.append(np.load(name)[:, np.newaxis])
                else:
                    data_list.append(np.load(name))
            vars(self)[dt + "_data"] = np.concatenate(data_list, axis=0)

        # Correct velocity
        for i in range(len(self.action_data)):
            rotation_rob_matrix = quaternion.as_rotation_matrix(np.quaternion(*self.rotation_data[i]))
            self.vel_data[i] = np.matmul(rotation_rob_matrix.T, self.vel_data[i])


    def make_train_val_data(self):
        obs, labels = self._preprocess_data()

        n_train = int(len(obs) * self.config['trn_val_ratio'])
        trn_indeces = np.random.choice(range(len(obs)), n_train, replace=False)
        trn_mask = np.zeros(len(obs), dtype=np.bool)
        trn_mask[trn_indeces] = True
        val_indeces = np.arange(len(obs))[~trn_mask]

        self.obs_trn = obs[trn_indeces]
        self.labels_trn = labels[trn_indeces]
        self.obs_val = obs[val_indeces]
        self.labels_val = labels[val_indeces]

    def _preprocess_data(self):
        # Calculate deltas
        rotation_delta = self.rotation_data[2:, 2:3] - self.rotation_data[1:-1, 2:3]
        vel_delta = self.vel_data[2:, 1:3] - self.vel_data[1:-1, 1:3]
        angular_data = self.angular_data[2:, 2:3] - self.angular_data[1:-1, 2:3]

        obs = np.concatenate((self.vel_data[1:-1, 0:2],
                              self.angular_data[1:-1, 2:3],
                              self.action_data[:-2, :]), axis=1)
        labels = np.concatenate((rotation_delta,
                                 vel_delta,
                                 angular_data), axis=1)

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
            loss = self.criterion(y_pred, T.tensor(y_trn, dtype=T.float32))

            if t % 1000 == 999 and self.config["verbose"]:
                eval_loss = self.eval()
                print(f"step: {t}, trn_loss: {loss.item()}, tst_loss: {eval_loss}")

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.config["verbose"]:
            eval_loss = self.eval()
            print(f"Done training, eval loss: {eval_loss}")

    def eval(self):
        with T.no_grad():
            y_pred = self.NN.predict_batch(self.obs_val)
            loss = self.criterion(y_pred, T.tensor(self.labels_val, dtype=T.float32))
        return loss.item()

    def save_model(self, name):
        if not os.path.exists("models"):
            os.mkdir("models/")
        T.save(self.NN.state_dict(), f"models/{name}")

    def load_model(self, filename):
        try:
            self.NN.load_state_dict(T.load(f"models/{filename}"))
        except:
            print("Failed to load trained model")

if __name__=="__main__":
    with open("configs/model_training.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    fm = ForwardModelTrainer(config)
    fm.load_data()
    fm.train()
    fm.save_model("saved_model")
    fm.load_model("saved_model")