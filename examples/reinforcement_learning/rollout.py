from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    logp: float
    value: float
    terminated: bool
    truncated: bool
    next_state: np.ndarray


@dataclass(frozen=True)
class RolloutBatch:
    states: np.ndarray
    actions: np.ndarray
    old_logps: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    episode_returns: List[float]


def collect_trajectories(env, actor_critic, num_episodes=10):
    trajectories = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        trajectory = []
        while not done:
            action, logp, value = actor_critic.get_action_logp_value(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append(
                Transition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    logp=float(logp),
                    value=float(value),
                    terminated=terminated,
                    truncated=truncated,
                    next_state=next_state,
                )
            )
            state = next_state
        trajectories.append(trajectory)
    return trajectories


def compute_gae_and_returns(trajectory, actor_critic, gamma=0.99, gae_lambda=0.95):
    rewards = np.array([step.reward for step in trajectory], dtype=np.float32)
    values = np.array([step.value for step in trajectory], dtype=np.float32)
    advantages = np.zeros_like(values)

    bootstrap_value = 0.0
    last_step = trajectory[-1]
    if last_step.truncated and not last_step.terminated:
        # Time-limit truncation is not a terminal failure state, so bootstrap
        # from the critic instead of forcing the final value to zero.
        bootstrap_value = actor_critic.get_value(last_step.next_state)

    gae = 0.0
    for t in reversed(range(len(trajectory))):
        if t == len(trajectory) - 1:
            next_value = bootstrap_value
            next_nonterminal = 0.0 if trajectory[t].terminated else 1.0
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def build_rollout_batch(trajectories, actor_critic, gamma=0.99, gae_lambda=0.95):
    states = []
    actions = []
    old_logps = []
    returns = []
    advantages = []
    episode_returns = []

    for trajectory in trajectories:
        adv, ret = compute_gae_and_returns(
            trajectory,
            actor_critic,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        episode_returns.append(sum(step.reward for step in trajectory))
        for step, step_adv, step_ret in zip(trajectory, adv, ret):
            states.append(step.state)
            actions.append(step.action)
            old_logps.append(step.logp)
            advantages.append(step_adv)
            returns.append(step_ret)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    old_logps = np.asarray(old_logps, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)
    advantages = np.asarray(advantages, dtype=np.float32)
    # Normalize advantages once per rollout to keep PPO updates well scaled.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return RolloutBatch(
        states=states,
        actions=actions,
        old_logps=old_logps,
        returns=returns,
        advantages=advantages,
        episode_returns=episode_returns,
    )
