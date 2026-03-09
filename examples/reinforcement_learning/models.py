import numpy as np
import eazygrad as ez
import eazygrad.nn as nn


def _build_mlp(input_dim, output_dim, hidden_dim, n_layer):
    layers = nn.ModuleList()
    layers.append(nn.Linear(n_in=input_dim, n_out=hidden_dim))
    for _ in range(n_layer - 1):
        layers.append(nn.Linear(n_in=hidden_dim, n_out=hidden_dim))
    layers.append(nn.Linear(n_in=hidden_dim, n_out=output_dim))
    return layers


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layer=2):
        super().__init__()
        self.layers = _build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layer=n_layer,
        )

    def forward(self, x):
        y = x
        for layer in self.layers[:-1]:
            y = ez.relu(layer(y))
        return self.layers[-1](y)


class Actor(MLP):
    def __init__(self, state_dim, act_dim, h_dim=256, n_layer=2):
        super().__init__(
            input_dim=state_dim,
            output_dim=act_dim,
            hidden_dim=h_dim,
            n_layer=n_layer,
        )


class Critic(MLP):
    def __init__(self, state_dim, h_dim=256, n_layer=2):
        super().__init__(
            input_dim=state_dim,
            output_dim=1,
            hidden_dim=h_dim,
            n_layer=n_layer,
        )


class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, h_dim=256, n_layer=2):
        super().__init__()
        self.actions = np.arange(act_dim)

        # Actor and critic init
        self.actor = Actor(state_dim=state_dim, act_dim=act_dim, h_dim=h_dim, n_layer=n_layer)
        self.critic = Critic(state_dim=state_dim, h_dim=h_dim, n_layer=n_layer)

    @ez.no_grad
    def get_action(self, state):
        action, _, _ = self.get_action_logp_value(state)
        return action

    @ez.no_grad
    def get_action_logp_value(self, state):
        state_t = ez.from_numpy(state).unsqueeze(0)
        logits = self.actor(state_t)
        probs = ez.softmax(logits, dim=-1).squeeze(0).numpy()
        action = int(np.random.choice(self.actions, p=probs))
        logp = float(self.get_logprob(logits, action).numpy())
        value = float(self.critic(state_t).squeeze(0).squeeze(0).numpy())
        return action, logp, value

    def get_logprob(self, logits, action: int):
        if len(logits.shape) == 2 and logits.shape[0] != 1:
            raise ValueError("get_logprob expects logits with batch size 1 (shape [1, act_dim]).")

        logZ = ez.logsumexp(logits, dim=-1).squeeze(0)
        logp = logits.squeeze(0)[action] - logZ
        return logp

    def get_value(self, state):
        state_t = ez.from_numpy(state).unsqueeze(0)
        v = self.critic(state_t)
        return float(v.squeeze(0).squeeze(0).numpy())

    def forward(self, state, action: int):
        s = ez.from_numpy(state).unsqueeze(0)
        logits = self.actor(s)
        value = self.critic(s).squeeze(0).squeeze(0)
        logp = self.get_logprob(logits, action)
        return logp, value
