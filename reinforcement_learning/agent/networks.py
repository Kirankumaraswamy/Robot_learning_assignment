import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple, List

"""
CartPole network
"""

class MLP(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=400):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)

class CNN(nn.Module):

    def __init__(self, n_classes=5, history_length=0):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        channel, width, height = (1+history_length, 96, 96)
        self.conv1 = nn.Conv2d(1+history_length, out_channels=32, kernel_size=(8, 8), stride=4, padding=1)
        width, height = calculate_conv_output([width, height], [8, 8], stride=[4, 4], padding=[1, 1])
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2, padding=1)
        width, height = calculate_conv_output([width, height], [4, 4], [2, 2], [1, 1])
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        width, height = calculate_conv_output([width, height], [3, 3], [1, 1], [1, 1])
        self.batch3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * width * height, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        # TODO: compute forward pass
        x = self.batch1(F.relu(self.conv1(x)))
        x = self.batch2(F.relu(self.conv2(x)))
        x = self.batch3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        return x

def calculate_conv_output(input_shape: List[int], kernel_size: List[int],
                          stride: List[int], padding: List[int]) -> List[int]:
    """Calculate the output shape of a convolution given the input shape and the convolution parameters.
    The input and output lists will all be of the same length N for an N-dimensional convolution.
    Args:
        input_shape: Input dimension sizes
        kernel_size: Kernel size per dimension
        stride: Stride per dimension
        padding: Padding per dimension
    Returns:
        Convolution output shape
    """
    w2 = (input_shape[0] - kernel_size[0] + 2 * padding[0]) / stride[0] + 1
    h2 = (input_shape[1] - kernel_size[1] + 2 * padding[1]) / stride[1] + 1

    output = (int(w2), int(h2))
    return output
