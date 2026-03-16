from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    critic_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    n_iter: int = 30
    rollout_episodes: int = 10
    ppo_epochs: int = 4
    minibatch_size: int = 64
