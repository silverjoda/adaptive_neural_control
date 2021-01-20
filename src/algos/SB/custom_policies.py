from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as T
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        history_length: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.feature_dim = feature_dim
        self.history_length = history_length
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net_tc = TemporalConvNet(num_inputs=self.feature_dim,
                                          num_channels=(16, 16, 16),
                                          kernel_size=2,
                                          dropout=0.0)


        self.policy_l1 = nn.Sequential(nn.Linear(self.feature_dim, 32), nn.ReLU())
        self.policy_l2 = nn.Sequential(nn.Linear(32 + 16, last_layer_dim_pi), nn.ReLU())

        self.value_net_tc = TemporalConvNet(num_inputs=self.feature_dim,
                                          num_channels=(16, 16, 16),
                                          kernel_size=2,
                                          dropout=0.0)

        self.value_l1 = nn.Sequential(nn.Linear(self.feature_dim, 32), nn.ReLU())
        self.value_l2 = nn.Sequential(nn.Linear(32 + 16, last_layer_dim_vf), nn.ReLU())


    def forward(self, features: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        features_conv = features.view((-1, self.history_length, self.feature_dim))
        features_conv = features_conv.transpose(2, 1)
        features_last_step = features[:, -self.feature_dim:]

        # FF of last obs step
        pi_l1 = self.policy_l1(features_last_step)
        vf_l1 = self.value_l1(features_last_step)

        # Conv of the whole thing
        policy_conv_features = self.policy_net_tc(features_conv)[:,:,-1]
        value_conv_features = self.value_net_tc(features_conv)[:,:,-1]

        # Combination of both
        pi_l2 = self.policy_l2(T.cat([pi_l1, policy_conv_features], dim=1))
        vf_l2 = self.value_l2(T.cat([vf_l1, value_conv_features], dim=1))

        return pi_l2, vf_l2

def customActorCriticPolicyWrapper(feature_dim, history_length):
    class CustomActorCriticPolicy(ActorCriticPolicy):
        def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
        ):

            super(CustomActorCriticPolicy, self).__init__(
                observation_space,
                action_space,
                lr_schedule,
                net_arch,
                activation_fn,
                # Pass remaining arguments to base class
                *args,
                **kwargs,
            )

            self.history_length = history_length
            self.feature_dim = feature_dim
            self.ortho_init = False

        def _build_mlp_extractor(self) -> None:
            self.mlp_extractor = CustomNetwork(feature_dim, history_length)

    return CustomActorCriticPolicy