import torch
import gym
import numpy as np
import pybullet_envs
import argparse
import time
import random
import tqdm
import wandb

from sac import SAC
from ReplayBuffer import ReplayBuffer
from utils import make_env, eval_policy


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--max_timesteps', type=int, default=1_000_000, help='total timesteps of the experiments')
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument('--alpha', type=float, default=0.2, help='Determines the relative importance of the entropy term')
    parser.add_argument('--lr', type=float, default=3e-4, help='the learning rate of the optimizer')
    parser.add_argument("--discount", default=0.99, type=float, help="Discount factor.")
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--fixed_alpha', action="store_true")

    args = parser.parse_args()

    seed = int(time.time())

    # SET SEED 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # CREATE ENVS
    env      = make_env(args.env, seed)
    eval_env = make_env(args.env, seed + 123)


    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])


    # INIT AGENT


    sac_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": args.discount,
        "tau": args.tau,
        "alpha": args.alpha,
        "lr": args.lr,
        "automatic_entropy_tuning": not args.fixed_alpha,
    }


    agent = SAC(**sac_kwargs)

    if args.fixed_alpha:
        algo_name = 'SAC_fixed_alpha'
    else:
        algo_name = 'SAC'

    experiment_name = f"{args.env}_{algo_name}_{int(time.time())}"


    # INIT REPLAY BUFFER
    replay_buffer = ReplayBuffer(state_dim, action_dim)


    # INIT LOGS
    wandb.init(project="rl_project", config=vars(args), name=experiment_name)


    # TRAIN
    state, done = env.reset(), False
    episode_timesteps = 0
    for t in tqdm.tqdm(range(args.max_timesteps)):

        episode_timesteps += 1
        action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action)
        
        done_float = float(done) if episode_timesteps < env._max_episode_steps else 0.

        replay_buffer.add(state, action, next_state, reward, done_float)
        
        state = next_state

        if done:
            state, done = env.reset(), False
            episode_timesteps = 0

        # update policy per data point
        agent_update_info = agent.train(replay_buffer.sample(args.batch_size))
        wandb.log({"train/": agent_update_info})

        # Evaluate episode
        if t % args.eval_freq == 0:
            eval_info = eval_policy(agent, eval_env)
            eval_info.update({'timesteps': t})
            print(f"Time steps: {t}, Eval_info: {eval_info}")
            wandb.log({"eval/": eval_info}) 






    if args.save_model:
        agent.save(f"./{experiment_name}")

    env.close()