import torch
import torch.nn as nn
from models.sub_modules import *


class DampedNSVMLP(nn.Module):
    def __init__(self, nsv_mlp, equilibrium, damping=.01, reverse=False, **kwargs):
        super().__init__()

        self.nsv_mlp = nsv_mlp
        self.equilibrium = torch.tensor(equilibrium).float().cuda()
        self.damping = damping

        self.reverse = reverse
    
    def forward(self, x):

        output = self.nsv_mlp(x) + self.damping * (self.equilibrium - x)
        return -output if self.reverse else output


class DeeperNSVMLP(nn.Module):
    def __init__(self, nsv_dim, **kwargs):
        super().__init__()

        self.fc1 = nn.Linear(nsv_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, nsv_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        return x



class NSVMLP(nn.Module):
    def __init__(self, nsv_dim, **kwargs):
        super().__init__()

        self.fc1 = nn.Linear(nsv_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, nsv_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x