import torch 
from torch import nn
import torch.nn.functional as F

class PolicyNet(nn.Module):

    def __init__(self, state_dim=3,hidden_dim=128,action_dim=1,action_bound = 2.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,action_dim)
        self.action_bound = action_bound

    def forward(self,x):
        x = F.relu(self.fc1(x))
        mu = self.action_bound * torch.tanh(self.fc2(x))
        return mu

class QValueNet(nn.Module):

    def __init__(self, state_dim=3, hidden_dim=128, action_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim,1)

    def forward(self,x,a):
        cat = torch.cat([x,a],dim=-1)
        output = F.relu(self.fc1(cat))
        output = F.relu(self.fc2(output))
        output = self.fc_out(output)
        return output

