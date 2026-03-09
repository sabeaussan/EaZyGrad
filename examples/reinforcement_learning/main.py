import gymnasium as gym

from config import PPOConfig
from models import ActorCritic
from trainer import train


def main():
    config = PPOConfig()
    env = gym.make("CartPole-v1")
    try:
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        actor_critic = ActorCritic(state_dim=state_dim, act_dim=act_dim)
        train(env, actor_critic, config=config)
    finally:
        env.close()


if __name__ == "__main__":
    main()
