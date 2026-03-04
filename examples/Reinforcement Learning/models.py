import numpy as np
import eazygrad as ez
import eazygrad.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, h_dim=256, n_layer=2):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(n_in=state_dim, n_out=h_dim))
        for _ in range(n_layer - 1):
            self.net.append(nn.Linear(n_in=h_dim, n_out=h_dim))
        self.net.append(nn.Linear(n_in=h_dim, n_out=act_dim))

    def forward(self, x):
        y = x
        for i in range(len(self.net) - 1):
            y = ez.relu(self.net[i](y))
        return self.net[-1](y)  # logits [B, act_dim]


class Critic(nn.Module):
    def __init__(self, state_dim, h_dim=256, n_layer=2):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(n_in=state_dim, n_out=h_dim))
        for _ in range(n_layer - 1):
            self.net.append(nn.Linear(n_in=h_dim, n_out=h_dim))
        self.net.append(nn.Linear(n_in=h_dim, n_out=1))

    def forward(self, x):
        y = x
        for i in range(len(self.net) - 1):
            y = ez.relu(self.net[i](y))
        return self.net[-1](y)  # [B, 1]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, h_dim=256, n_layer=2):
        super().__init__()
        self.actions = np.arange(act_dim)

        # Actor and critic init
        self.actor = Actor(state_dim=state_dim, act_dim=act_dim, h_dim=h_dim, n_layer=n_layer)
        self.critic = Critic(state_dim=state_dim, h_dim=h_dim, n_layer=n_layer)

    @ez.no_grad
    def get_action(self, state):
        state_t = ez.from_numpy(state).unsqueeze(0)   # [1, state_dim]
        logits = self.actor(state_t)                  # [1, act_dim]
        probs = ez.softmax(logits, dim=-1).squeeze(0).numpy()
        return int(np.random.choice(self.actions, p=probs))

    def get_logprob(self, logits, action: int):
        """
        logits: [1, act_dim] or [B, act_dim]
        If [B, act_dim], 'action' should be int only when B==1.
        """
        if len(logits.shape) == 2 and logits.shape[0] != 1:
            raise ValueError("get_logprob expects logits with batch size 1 (shape [1, act_dim]).")

        logZ = ez.logsumexp(logits, dim=-1).squeeze(0)     # scalar
        logp = logits.squeeze(0)[action] - logZ            # scalar
        return logp

    def get_value(self, state):
        """
        Infer V(s) from critic. Returns a Python float.
        """
        state_t = ez.from_numpy(state).unsqueeze(0)        # [1, state_dim]
        v = self.critic(state_t)                           # [1, 1]
        return float(v.squeeze(0).squeeze(0).numpy())

    def forward(self, state, action: int):
        """
        Differentiable evaluation for training:
          state: numpy array shape [state_dim]
          action: int
        Returns:
          logp (Tensor scalar), value (Tensor scalar)
        """
        s = ez.from_numpy(state).unsqueeze(0)              # [1, state_dim]
        logits = self.actor(s)                             # [1, act_dim]
        value = self.critic(s).squeeze(0).squeeze(0)       # scalar Tensor
        logp = self.get_logprob(logits, action)            # scalar Tensor
        return logp, value