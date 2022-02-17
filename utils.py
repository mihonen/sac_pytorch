
import numpy as np
import gym
import pybullet_envs


def eval_policy(policy, eval_env, eval_episodes=10):
    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment

    rewards = np.zeros(eval_episodes)

    for i in range(eval_episodes):
        state, done = eval_env.reset(), False
        episode_reward = 0
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            episode_reward += reward

        rewards[i] = episode_reward


    return {'returns': np.mean(rewards)}

def make_env(name, seed: int):
    env = gym.make(name)
    env.seed(seed)
    env.action_space.seed(seed)
    return env



