import eazygrad as ez
import eazygrad.nn as nn
import gymnasium as gym
import numpy as np



class Actor(nn.Module):

    def __init__(self, state_dim, act_dim, h_dim=256, n_layer=2):
        self.net = nn.ModuleList()
        self.actions = np.arange(act_dim)
        self.net.append(nn.Linear(n_in=state_dim, n_out=h_dim))
        for _ in range(n_layer - 1):
            self.net.append(nn.Linear(n_in=h_dim, n_out=h_dim))
        self.net.append(nn.Linear(n_in=h_dim, n_out=act_dim))

    def forward(self, x):
        y = x
        for i in range(len(self.net) - 1):
            y = ez.relu(self.net[i](y))
        return self.net[-1](y)

    @ez.no_grad
    def get_action(self, state):
        state = ez.from_numpy(state).unsqueeze(0)
        logits = self.forward(state)
        probs = ez.softmax(logits, dim=-1).squeeze(0).numpy()

        # sample using probs
        action = int(np.random.choice(self.actions, p=probs))
        return action
    
    def get_logprobs(self, logits, action):
        # log π(a|s) = logits[a] - logsumexp(logits)
        logZ = ez.logsumexp(logits, dim=-1)
        logp = logits.squeeze(0)[action] - logZ
        return logp

class Critic(nn.Module):

    def __init__(self, state_dim, h_dim=256, n_layer=2):
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(n_in=state_dim, n_out=h_dim))
        for _ in range(n_layer - 1):
            self.net.append(nn.Linear(n_in=h_dim, n_out=h_dim))
        self.net.append(nn.Linear(n_in=h_dim, n_out=1))

    def forward(self, x):
        y = x
        for i in range(len(self.net) - 1):
            y = ez.relu(self.net[i](y))
        return self.net[-1](y)



def collect_trajectories(env, policy, num_episode=10):
    """
      Collecte des trajectoires (épisodes) en suivant la politique donnée.
      On collecte l'équivalent de "num_episode" de donnée sur env.
    """
    trajectories = []

    # Collect a batch of trajectories
    for _ in range(num_episode):
        state, _ = env.reset()
        done = False
        trajectory = []

        while not done:
            action = policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, action, reward))
            state = next_state

        trajectories.append(trajectory)
    return trajectories

def compute_returns(rewards, gamma=0.99):
    """
      Calcul de la récompense cumulée avec un facteur de réduction gamma.
      "rewards" est une liste de récompenses.
    """
    G = 0 # G est un accumulateur
    cumulated_reward = [] # récompense cumulée
    for r in reversed(rewards):
        G = r + gamma * G
        cumulated_reward.insert(0, G)
    return cumulated_reward


def compute_policy_score(trajectories, gamma):
    """
      Calcul le score de la politique en utilisant les trajectoires (ou épisodes) récoltés.
    """
    score = 0
    all_rewards = []
    for trajectory in trajectories:
        states, actions, rewards = zip(*trajectory)
        returns = compute_returns(rewards, gamma=gamma)
        all_rewards.append(sum(rewards))
        returns = ez.tensor(returns)

        for t in range(len(trajectory)):
            state = ez.from_numpy(states[t]).unsqueeze(0)
            action_logits = policy(state)
            log_probs = policy.get_logprobs(action_logits, actions[t])
            score += -log_probs * returns[t]

    score /= len(trajectories)  # normalize by batch size
    return score, all_rewards

def train(env, policy, n_iter, n_episodes=3, gamma=0.99):
    # On définis l'optimizer (Adam)
    optimizer = ez.AdamW(policy.net.parameters(), lr=1e-3)

    # List pour afficher l'évolution à la fin de l'entrainement
    avg_rewards = []
    std_rewards = []

    best_avg_reward = -np.inf
    best_policy_state = None

    # Boucle d'entrainement
    for iteration in range(n_iter):
        trajectories = collect_trajectories(env, policy, num_episode=n_episodes)
        policy_score, all_rewards = compute_policy_score(trajectories, gamma)
        optimizer.zero_grad()
        policy_score.backward()
        optimizer.step()

        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)

        avg_rewards.append(avg_reward)
        std_rewards.append(std_reward)

        # Sauvegarde la version de la politique la plus performante
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            print(f"Nouveau record : {best_avg_reward:.2f} ! ")
            print("Sauvegarde de la politique ...")
            # best_policy_state = policy.state_dict()  # Sauvegarde les poids du model

        if (iteration + 1) % 5 == 0:
            print(f"Iteration {iteration + 1}, Average reward: {avg_reward:.2f}, Std: {std_reward:.2f}")

    # Affiche l'évolution de la récompense au cours de l'entrainement
    # plot_training_rewards(avg_rewards, std_rewards)
    return "ok"

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    policy = PolicyNetwork(state_dim=env.observation_space.shape[0], act_dim=env.action_space.n)
    num_iter = 500 # nombre d'iteration d'entrainement
    num_collect_episode = 2 # Nombre d'épisode à collecter avant l'update de la politique
    best_policy = train(env, policy, num_iter, num_collect_episode)