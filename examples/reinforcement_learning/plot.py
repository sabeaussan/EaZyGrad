from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import eazygrad as ez
from eazygrad.grad import dag


def restore_model_state(model, state):
    for param, saved in zip(model.parameters(), state):
        param._array.flags.writeable = True
        param._array[...] = saved


def save_reward_plot(avg_rewards, std_rewards, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iterations = np.arange(1, len(avg_rewards) + 1)
    avg_rewards = np.asarray(avg_rewards, dtype=np.float32)
    std_rewards = np.asarray(std_rewards, dtype=np.float32)

    # Plot both the mean return and its spread across rollout episodes.
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(iterations, avg_rewards, linewidth=2, label="average return")
    ax.fill_between(iterations, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, label="std")
    ax.set_title("PPO training on CartPole-v1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    path = output_dir / "reward_curve.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _greedy_action(actor_critic, state):
    prev_grad_state = dag.grad_enable
    dag.grad_enable = False
    try:
        # Use a greedy action for the rendered rollout so the saved GIF is stable.
        state_t = ez.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        logits = actor_critic.actor(state_t).numpy()[0]
    finally:
        dag.grad_enable = prev_grad_state
    return int(np.argmax(logits))


def save_policy_render(actor_critic, output_dir, env_name="CartPole-v1", max_steps=500):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rendering uses a dedicated env instance so the training env can stay headless.
    env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    episode_return = 0.0

    try:
        state, _ = env.reset(seed=0)
        frames.append(Image.fromarray(env.render()))
        for _ in range(max_steps):
            action = _greedy_action(actor_critic, state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            frames.append(Image.fromarray(env.render()))
            if terminated or truncated:
                break
    finally:
        env.close()

    path = output_dir / "best_policy.gif"
    # Save as GIF because it is simple to inspect without extra tooling.
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    return path, episode_return
