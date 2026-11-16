from torch import nn

class PolicyNet(nn.Module):
    def __init__(self,state_dim=4,hidden_dim=128,action_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        probs = self.model(x)
        return probs

class ValueNet(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,x):
        return self.model(x)