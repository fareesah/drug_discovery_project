import torch
import torch.nn as nn

class AffinityPredictor(nn.Module):
    def __init__(self, input_dim=1024):
        super(AffinityPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
