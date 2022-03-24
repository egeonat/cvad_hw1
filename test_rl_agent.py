import argparse
import os

import torch
import yaml

from carla_env.env import Env
from rl.td3 import TD3


def parse_args():
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Evaluate an RL agent")
    parser.add_argument("--config", type=str, default="td3.yaml")
    parser.add_argument("--ckpt", type=str,
                        default="last_td3.ckpt")
    parser.add_argument("--attempts", type=int, default=100)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        ckpt_path = os.path.join("checkpoints", "rl", args.ckpt)
        alg = TD3(env, config)
        alg.load_ckpt(ckpt_path)
        print(f"Loaded policy from best run at step: {alg.step}")
        with torch.no_grad():
            ep_step, ep_return, ep_mean_reward, t_hist = alg.test_agent(args.attempts)
    print(f"Evaluation complete with {args.attempts} runs. Causes for termination:")
    for cause, count in t_hist.items():
        print(f"Cause: {cause}. {count}/{args.attempts}")


if __name__ == "__main__":
    main()
