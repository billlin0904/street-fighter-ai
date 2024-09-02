from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
import gym

class CustomCnnLstmExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCnnLstmExtractor, self).__init__(observation_space, features_dim)

        # CNN Layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the CNN output size to feed into LSTM
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # LSTM Layer
        self.lstm = nn.LSTM(n_flatten, features_dim, batch_first=True)

    def forward(self, observations: th.Tensor):
        cnn_out = self.cnn(observations)
        cnn_out = cnn_out.unsqueeze(0)  # Add sequence dimension
        lstm_out, _ = self.lstm(cnn_out)
        return lstm_out.squeeze(0)  # Remove sequence dimension

# Custom policy with the LSTM feature extractor
class CustomLstmPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomLstmPolicy, self).__init__(*args, **kwargs,
                                               features_extractor_class=CustomCnnLstmExtractor,
                                               features_extractor_kwargs=dict(features_dim=256))
