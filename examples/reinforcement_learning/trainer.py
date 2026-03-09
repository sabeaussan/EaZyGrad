import numpy as np

import eazygrad as ez

from config import PPOConfig
from rollout import RolloutBatch, build_rollout_batch, collect_trajectories


def compute_entropy(actor_critic, state):
    logits = actor_critic.actor(ez.from_numpy(state).unsqueeze(0))
    probs = ez.softmax(logits, dim=-1).squeeze(0)
    return -(probs * ez.log(probs + 1e-8)).sum()


def compute_ppo_losses(actor_critic, batch: RolloutBatch, indices, config):
    actor_loss = ez.tensor(0.0, requires_grad=True)
    critic_loss = ez.tensor(0.0, requires_grad=True)
    entropy_loss = ez.tensor(0.0, requires_grad=True)

    for idx in indices:
        state = batch.states[idx]
        action = int(batch.actions[idx])
        old_logp = ez.tensor(batch.old_logps[idx], requires_grad=False)
        advantage = ez.tensor(batch.advantages[idx], requires_grad=False)
        target_return = ez.tensor(batch.returns[idx], requires_grad=False)

        logp, value = actor_critic(state, action)
        ratio = ez.exp(logp - old_logp)
        clipped_ratio = ez.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
        actor_loss = actor_loss + (-ez.min(ratio * advantage, clipped_ratio * advantage))

        value_error = value - target_return
        critic_loss = critic_loss + (value_error * value_error)
        entropy_loss = entropy_loss - compute_entropy(actor_critic, state)

    inv_batch = np.float32(1.0 / len(indices))
    actor_loss = actor_loss * inv_batch
    critic_loss = critic_loss * inv_batch
    entropy_loss = entropy_loss * inv_batch
    total_loss = (
        actor_loss
        + config.critic_coef * critic_loss
        + config.entropy_coef * entropy_loss
    )
    return total_loss, actor_loss, critic_loss, entropy_loss


def train(env, actor_critic, config: PPOConfig):
    optimizer = ez.AdamW(actor_critic.parameters(), lr=config.lr)
    avg_rewards = []
    std_rewards = []
    best_avg_reward = -np.inf

    for it in range(config.n_iter):
        trajectories = collect_trajectories(
            env,
            actor_critic,
            num_episodes=config.rollout_episodes,
        )
        batch = build_rollout_batch(
            trajectories,
            actor_critic,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        num_steps = len(batch.states)
        permutation = np.arange(num_steps)

        last_actor_loss = None
        last_critic_loss = None
        last_entropy_loss = None

        for _ in range(config.ppo_epochs):
            np.random.shuffle(permutation)
            for start in range(0, num_steps, config.minibatch_size):
                indices = permutation[start:start + config.minibatch_size]
                if len(indices) == 0:
                    continue
                total_loss, actor_loss, critic_loss, entropy_loss = compute_ppo_losses(
                    actor_critic,
                    batch,
                    indices,
                    config,
                )
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                last_actor_loss = actor_loss
                last_critic_loss = critic_loss
                last_entropy_loss = entropy_loss

        avg_r = float(np.mean(batch.episode_returns))
        std_r = float(np.std(batch.episode_returns))
        avg_rewards.append(avg_r)
        std_rewards.append(std_r)

        if avg_r > best_avg_reward:
            best_avg_reward = avg_r
            print(f"New best avg return: {best_avg_reward:.2f}")

        if (it + 1) % 5 == 0:
            print(
                f"Iter {it+1:4d} | avg return {avg_r:6.2f} +- {std_r:5.2f} "
                f"| actor_loss {last_actor_loss.numpy() if last_actor_loss is not None else 'n/a'} "
                f"| critic_loss {last_critic_loss.numpy() if last_critic_loss is not None else 'n/a'} "
                f"| entropy_loss {last_entropy_loss.numpy() if last_entropy_loss is not None else 'n/a'}"
            )

    return avg_rewards, std_rewards
