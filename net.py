import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Create and run the encoder network
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 20 * 20, 1)

    def forward(self, frame):
        """
        :param frame: the current frame
        :return: The encoder value of the frame
        """
        frame = F.relu(self.conv1(frame))
        frame = frame.view(-1, 16 * 20 * 20)
        frame = self.fc1(frame)
        return frame


class Transition(nn.Module):
    """
    Create and run the transition network.
    """
    def __init__(self):
        super(Transition, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, frame):
        """
        :param frame: the current frame
        :return: The encoder value of the frame
        """
        frame = F.relu(self.fc1(frame))
        frame = self.fc2(frame)
        return frame