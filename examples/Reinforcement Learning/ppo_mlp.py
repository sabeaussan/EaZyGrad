import eazygrad as ez
import gymnasium as gym
import numpy as np
from models import ActorCritic

def collect_trajectories(env, actor_critic, num_episode=10):
    trajectories = []
    for _ in range(num_episode):
        state, _ = env.reset()
        done = False
        trajectory = []
        while not done:
            action = actor_critic.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, action, float(reward)))
            state = next_state
        trajectories.append(trajectory)
    return trajectories


def compute_returns(rewards, gamma=0.99):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out  # python list


def compute_actor_critic_losses(trajectories, actor_critic, gamma=0.99, critic_coef=0.5):
    """
    Returns:
      total_loss (ez Tensor scalar),
      actor_loss (ez Tensor scalar),
      critic_loss (ez Tensor scalar),
      episode_returns (list[float])
    """
    actor_loss = ez.tensor(0.0)
    critic_loss = ez.tensor(0.0)
    episode_returns = []

    # We’ll normalize by number of episodes (and optionally by timesteps)
    total_steps = 0

    for traj in trajectories:
        states, actions, rewards = zip(*traj)
        episode_returns.append(sum(rewards))

        returns = compute_returns(rewards, gamma=gamma)          # list length T
        returns_t = ez.tensor(returns)                           # [T] tensor (constants)

        for t in range(len(traj)):
            Gt = returns_t[t]                                     # scalar

            # Critic prediction V(s)
            logp, value = actor_critic(states[t], actions[t])

            # Advantage A = G - V
            A = Gt - value
            A_detached = A.detach() 
            actor_loss = actor_loss + (-logp * A_detached)

            # Critic: MSE(V, G)
            diff = (value - Gt)
            critic_loss = critic_loss + (diff * diff)

            total_steps += 1

    # Average losses (helps make lr less sensitive to batch size / trajectory length)
    if total_steps > 0:
        actor_loss = actor_loss * (1.0 / total_steps)
        critic_loss = critic_loss * (1.0 / total_steps)

    total_loss = critic_coef * critic_loss + actor_loss 
    return total_loss, actor_loss, critic_loss, episode_returns


def train(env, actor_critic, n_iter, n_episodes=10, gamma=0.99, lr=1e-3, critic_coef=0.5):
    # Optimize BOTH actor and critic params
    optimizer = ez.AdamW(actor_critic.parameters(), lr=lr)

    avg_rewards = []
    std_rewards = []

    best_avg_reward = -np.inf

    for it in range(n_iter):
        trajectories = collect_trajectories(env, actor_critic, num_episode=n_episodes)

        total_loss, a_loss, c_loss, ep_returns = compute_actor_critic_losses(
            trajectories, actor_critic, gamma=gamma, critic_coef=critic_coef
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        avg_r = float(np.mean(ep_returns))
        std_r = float(np.std(ep_returns))
        avg_rewards.append(avg_r)
        std_rewards.append(std_r)

        if avg_r > best_avg_reward:
            best_avg_reward = avg_r
            print(f"New best avg return: {best_avg_reward:.2f}")

        if (it + 1) % 5 == 0:
            print(
                f"Iter {it+1:4d} | avg return {avg_r:6.2f} ± {std_r:5.2f} "
                f"| actor_loss {a_loss.numpy() if hasattr(a_loss,'numpy') else a_loss} "
                f"| critic_loss {c_loss.numpy() if hasattr(c_loss,'numpy') else c_loss}"
            )

    return avg_rewards, std_rewards


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    actor_critic = ActorCritic(state_dim=state_dim, act_dim=act_dim)
    train(env, actor_critic, n_iter=500, n_episodes=10, gamma=0.99, lr=1e-3, critic_coef=0.5)