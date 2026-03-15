import numpy as np

import eazygrad as ez

from config import PPOConfig
from rollout import RolloutBatch, build_rollout_batch, collect_trajectories


def compute_ppo_losses(actor_critic, batch: RolloutBatch, indices, config):
    states = batch.states[indices]
    actions = batch.actions[indices]
    old_logps = ez.from_numpy(batch.old_logps[indices])
    advantages = ez.from_numpy(batch.advantages[indices])
    target_returns = ez.from_numpy(batch.returns[indices])

    logits, logp, values = actor_critic.evaluate_actions(states, actions)
    ratio = ez.exp(logp - old_logps)
    clipped_ratio = ez.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
    # PPO keeps the smaller of the unclipped and clipped objectives to prevent
    # overly large policy updates.
    surrogate = ez.min(ratio * advantages, clipped_ratio * advantages)
    actor_loss = -surrogate.mean()

    value_error = values - target_returns
    critic_loss = (value_error * value_error).mean()

    probs = ez.softmax(logits, dim=-1)
    # Encourage exploration by penalizing overly sharp action distributions.
    entropy = -(probs * ez.log(probs + 1e-8)).sum(dim=-1).mean()

    total_loss = actor_loss + config.critic_coef * critic_loss - config.entropy_coef * entropy
    return total_loss, actor_loss, critic_loss, entropy


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
                # Reuse the same rollout for several epochs, but reshuffle the
                # minibatches each time to decorrelate updates.
                indices = permutation[start:start + config.minibatch_size]
                if len(indices) == 0:
                    continue
                total_loss, actor_loss, critic_loss, entropy = compute_ppo_losses(
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
                last_entropy_loss = entropy

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
                f"| entropy {last_entropy_loss.numpy() if last_entropy_loss is not None else 'n/a'}"
            )

    return avg_rewards, std_rewards
