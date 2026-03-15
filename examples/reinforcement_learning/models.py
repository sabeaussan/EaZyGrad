import numpy as np
import eazygrad as ez
import eazygrad.nn as nn


def _build_mlp(input_dim, output_dim, hidden_dim, n_layer):
    layers = nn.ModuleList()
    # Build hidden blocks explicitly so examples can inspect the layer list.
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
        # Sample in NumPy space because action selection is intentionally kept
        # outside the autograd graph.
        action = int(np.random.choice(self.actions, p=probs))
        logp = float(self.get_logprob(logits, action).numpy())
        value = float(self.critic(state_t).squeeze(0).squeeze(0).numpy())
        return action, logp, value

    def get_logprob(self, logits, action):
        if logits.ndim != 2:
            raise ValueError(f"get_logprob expects logits with shape [batch, act_dim], got {logits.shape}.")

        # Compute log-probabilities via logsumexp for numerical stability.
        logZ = ez.logsumexp(logits, dim=-1)
        batch_size = logits.shape[0]

        if np.isscalar(action):
            if batch_size != 1:
                raise ValueError("Scalar action only supported when batch size is 1.")
            selected_logits = logits[0, int(action)]
            return selected_logits - logZ.squeeze(0)

        action = np.asarray(action, dtype=np.int64)
        if action.shape != (batch_size,):
            raise ValueError(f"Expected actions with shape ({batch_size},), got {action.shape}.")

        batch_index = np.arange(batch_size, dtype=np.int64)
        selected_logits = logits[batch_index, action]
        return selected_logits - logZ

    def get_value(self, state):
        state_t = ez.from_numpy(state).unsqueeze(0)
        v = self.critic(state_t)
        return float(v.squeeze(0).squeeze(0).numpy())

    def evaluate_actions(self, states, actions):
        # PPO repeatedly re-evaluates old actions under the current policy.
        states_t = ez.from_numpy(np.asarray(states, dtype=np.float32))
        logits = self.actor(states_t)
        values = self.critic(states_t).reshape(-1)
        logp = self.get_logprob(logits, actions)
        return logits, logp, values

    def forward(self, state, action):
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim == 1:
            _, logp, values = self.evaluate_actions(
                states=np.expand_dims(state_array, axis=0),
                actions=np.array([action], dtype=np.int64),
            )
            return logp.squeeze(0), values.squeeze(0)

        _, logp, values = self.evaluate_actions(states=state_array, actions=action)
        return logp, values
