import torch.nn as nn
from torch.autograd import Variable
import torch


class car_detection_net(nn.Module):
    def __init__(self):
        super(car_detection_net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # (3,60,30) => (16,30,15)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # (16,30,15) => (64,15,7)

        self.fc3 = nn.Linear(64*15*7, 60)
        self.nonLine4 = nn.ReLU()
        self.fc5 = nn.Linear(60, 2)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        x = self.nonLine4(x)
        x = self.fc5(x)
        x = self.out(x)

        return x
