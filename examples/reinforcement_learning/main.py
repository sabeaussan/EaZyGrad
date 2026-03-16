from pathlib import Path

import gymnasium as gym

from config import PPOConfig
from models import ActorCritic
from plot import restore_model_state, save_policy_render, save_reward_plot
from trainer import train


def main():
    config = PPOConfig()
    output_dir = Path(__file__).resolve().parent / "figures"
    # The training env stays lightweight; rendering is handled separately after
    # training so rollout collection is not slowed down.
    env = gym.make("CartPole-v1")
    try:
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        actor_critic = ActorCritic(state_dim=state_dim, act_dim=act_dim)
        avg_rewards, std_rewards, best_state = train(env, actor_critic, config=config)
    finally:
        env.close()

    # Visualize the checkpoint that achieved the best average rollout return.
    restore_model_state(actor_critic, best_state)
    reward_plot_path = save_reward_plot(avg_rewards, std_rewards, output_dir)
    render_path, render_return = save_policy_render(actor_critic, output_dir)
    print(f"Saved reward curve to {reward_plot_path}")
    print(f"Saved best-policy rollout to {render_path} (return={render_return:.2f})")


if __name__ == "__main__":
    main()
