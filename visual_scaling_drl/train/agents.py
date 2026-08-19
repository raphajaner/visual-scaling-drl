"""Actor-Critic and PPO agent network definitions."""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from visual_scaling_drl.nn.utils import layer_init_orthogonal
from visual_scaling_drl.nn.encoder import encoder_factory


class ActorCriticAgent(nn.Module):
    """Base Actor-Critic agent wrapping visual encoders with policy and value heads."""
    def __init__(
            self,
            envs,
            encoder_type='impoola',
            width_scale=1, out_features=256, cnn_filters=(16, 32, 32), kernel_size=3, activation='relu',
            pooling_type=False, pooling_layer_kernel_size=1,
            use_dropout=False,
            use_moe=False,
            use_relu_after_last_conv=True,
            strong_downsample_first=False,
            use_drqv2_style=False
    ):
        super().__init__()

        # Encode input images (input as int8, conversion to float32 is done in the encoder forward pass)
        encoder, out_features = encoder_factory(
            encoder_type=encoder_type,
            envs=envs,
            width_scale=width_scale, out_features=out_features, cnn_filters=cnn_filters, kernel_size=kernel_size,
            activation=activation,
            pooling_type=pooling_type, pooling_layer_kernel_size=pooling_layer_kernel_size,
            use_dropout=use_dropout,
            use_moe=use_moe,
            use_relu_after_last_conv=use_relu_after_last_conv,
            strong_downsample_first=strong_downsample_first,
            use_drqv2_style=use_drqv2_style
        )
        self.encoder = encoder
        self.out_features = out_features

        # Actor head
        actor = nn.Linear(out_features, envs.single_action_space.n)
        self.actor = layer_init_orthogonal(actor, std=0.01)

        # Critic head
        critic = nn.Linear(out_features, 1)
        self.critic = layer_init_orthogonal(critic, std=1.0)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.actor(hidden), self.critic(hidden)

    def get_value(self, x):
        return self.forward(x)[1]

    def get_pi(self, x):
        return Categorical(logits=self.forward(x)[0])

    def get_action(self, x, deterministic=False):
        pi = self.get_pi(x)
        return pi.sample() if not deterministic else pi.mode

    def get_action_and_value(self, x, action=None):
        raise NotImplementedError


class PPOAgent(ActorCriticAgent):
    """PPO agent implementing action sampling and log probability computation."""
    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        pi = Categorical(logits=logits)
        if action is None:
            action = pi.sample()
        return action, pi.log_prob(action), pi.entropy(), value, pi.logits

    def get_pi_and_value(self, x):
        logits, value = self.forward(x)
        return Categorical(logits=logits), value

