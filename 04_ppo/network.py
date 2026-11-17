from torch import nn
import torch
import torch.nn.functional as F

class PolicyNet(nn.Module):

    def __init__(self, state_dim=4,hidden_dim=128,action_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        return self.model(x)

class ValueNet(nn.Module):

    def __init__(self, state_dim=4,hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,x):
        return self.model(x)

class PolicyNetContinuous(nn.Module):

    def __init__(self, state_dim=3, hidden_dim=128, action_dim=1, action_bound=2.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim,action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_bound = action_bound

    def forward(self,x):
        x = F.relu(self.fc1(x))
        mu = self.action_bound * torch.tanh(self.fc_mu(x))
        log_std = torch.clamp(self.log_std, -4.0, 1.0)
        std = torch.exp(log_std).expand_as(mu)
        return mu,std
