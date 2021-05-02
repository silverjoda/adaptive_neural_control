import os
import glob

import matplotlib.pyplot as plt
import yaml
import numpy as np
import torch as T
import torch.nn as nn
import torch.functional as F
import quaternion

T.set_num_threads(1)

class ForwardNet(nn.Module):
    def __init__(self, config):
        super(ForwardNet, self).__init__()
        self.config = config

        self.l1 = nn.Linear(self.config["input_dim"], self.config["hidden_dim"])
        self.l2 = nn.Linear(self.config["hidden_dim"], self.config["hidden_dim"])
        self.l3 = nn.Linear(self.config["hidden_dim"], self.config["output_dim"])

        self.non_linearity = eval(self.config["non_linearity"])()

        #for p in self.parameters():
        #   p.register_hook(lambda grad: T.clamp(grad, -config["policy_grad_clip_value"], config["policy_grad_clip_value"]))

    def forward(self, x):
        feat1 = self.non_linearity(self.l1(x))
        feat2 = self.non_linearity(self.l2(feat1))
        out = self.l3(feat2)
        return out

    def predict(self, x):
        x = T.tensor(x, dtype=T.float32).unsqueeze(0)
        return self.forward(x)

    def predict_batch(self, x):
        return self.forward(x)

class ForwardModelTrainer:
    def __init__(self, config):
        self.config = config

        self.load_data()
        self.make_train_val_data()

        self.NN = ForwardNet(config)

        self.criterion = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.NN.parameters(),
                                 lr=self.config["learning_rate"],
                                 weight_decay=self.config["weight_decay"])

    def load_data(self):
        data_types_list = ["action", "angular", "position", "rotation", "timestamp", "vel"]
        for dt in data_types_list:
            data_list = []
            for name in glob.glob(f'data/{self.config["data_dir_name"]}/*{dt}.npy'):
                if dt == "timestamp":
                    data_list.append(np.load(name)[:, np.newaxis])
                else:
                    data_list.append(np.load(name))
            vars(self)[dt + "_data"] = np.concatenate(data_list, axis=0)

        # Correct velocity
        # for i in range(len(self.action_data)):
        #     rotation_rob_matrix = quaternion.as_rotation_matrix(np.quaternion(*self.rotation_data[i]))
        #     self.vel_data[i] = np.matmul(rotation_rob_matrix.T, self.vel_data[i])

        # Plot data

        plt.figure()
        plt.title("Actions")
        plt.plot(self.action_data[1000:1100, 0])
        plt.plot(self.action_data[1000:1100, 1])

        plt.figure()
        plt.title("Vels")
        plt.plot(self.vel_data[1000:1100, 0])
        plt.plot(self.vel_data[1000:1100, 1])
        plt.plot(self.vel_data[1000:1100, 2])

        plt.figure()
        plt.title("Angular vels")
        #plt.plot(t, self.angular_data[:, 0])
        #plt.plot(t, self.angular_data[:, 1])
        plt.plot(self.angular_data[1000:1100, 2])
        plt.show()

        # TODO: add low pass filtering on data to remove noise and replot to see the difference and tune the lp filter
        exit()

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
        clip_val = 2.
        vel_delta = np.clip(self.vel_data[0:-1, 0:2] - self.vel_data[1:, 0:2], -clip_val, clip_val) * 10
        angular_delta = np.clip(self.angular_data[0:-1, 2:3] - self.angular_data[1:, 2:3], -clip_val, clip_val) * 10

        n_data = len(vel_delta)
        n_hist = self.config["n_hist"]

        # Make history of obs
        vel_obs = np.zeros((n_data, 2 * n_hist))
        for i in range(n_hist - 1, n_data - 1):
            vel_obs[i, :] = self.vel_data[i - n_hist + 1:i + 1, 0:2].reshape(2 * n_hist)

        # Make history of ang obs
        angular_obs = np.zeros((n_data, 1 * n_hist))
        for i in range(n_hist - 1, n_data - 1):
            angular_obs[i, :] = self.angular_data[i - n_hist + 1:i + 1, 2:3].reshape(1 * n_hist)

        # Make history of actions obs
        action_obs = np.zeros((n_data, 2 * n_hist))
        for i in range(n_hist - 1, n_data - 1):
            action_obs[i, :] = self.action_data[i - n_hist + 1:i+1, 0:2].reshape(2 * n_hist)

        #action_obs = (np.array(self.action_data[0:-1, 0:2]) - 0.1) * 5
        #vel_obs = self.vel_data[0:-1, 0:2]
        #angular_obs = self.angular_data[0:-1, 2:3]

        obs = np.concatenate((vel_obs, angular_obs, action_obs), axis=1)
        labels = np.concatenate((vel_delta,
                                 angular_delta), axis=1)

        # Plot histograms
        #plt.hist(self.angular_data[:, 2:3], bins=100)
        # plt.hist(vel_delta[:, 1], bins=100)
        #plt.show()
        #exit()

        return obs, labels

    def get_batch(self, batchsize):
        trn_indeces = np.random.choice(range(len(self.obs_trn) - 1), batchsize, replace=False)
        obs_batch = self.obs_trn[trn_indeces]
        labels_batch = self.labels_trn[trn_indeces]
        return obs_batch, labels_batch

    def train(self):
        for t in range(self.config["n_trn_iters"]):
            x_trn, y_trn = self.get_batch(self.config["trn_batchsize"])

            x_trn_tensor = T.tensor(x_trn, dtype=T.float32)
            y_trn_tensor = T.tensor(y_trn, dtype=T.float32)

            y_pred = self.NN.predict_batch(x_trn_tensor)
            loss = self.criterion(y_trn_tensor, y_pred)

            if t % 1000 == 0 and self.config["verbose"]:
                eval_loss = self.eval()
                print(f"step: {t}, trn_loss: {loss.item()}, tst_loss: {eval_loss}")

                #rnd_idx = np.random.randint(0, self.config["trn_batchsize"])
                #print(x_trn[rnd_idx], y_trn[rnd_idx], y_pred[rnd_idx])

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.config["verbose"]:
            eval_loss = self.eval()
            print(f"Done training, eval loss: {eval_loss}")

    def eval(self):
        with T.no_grad():
            x_val_tensor = T.tensor(self.obs_val, dtype=T.float32)
            y_val_tensor = T.tensor(self.labels_val, dtype=T.float32)
            y_pred = self.NN.predict_batch(x_val_tensor)
            loss = self.criterion(y_pred, y_val_tensor)
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
    fm.train()
    fm.save_model("saved_model")
    fm.load_model("saved_model") # just to see if it works