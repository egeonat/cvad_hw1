import argparse
import os

import yaml

from carla_env.env import Env
from rl.td3 import TD3


def parse_args():
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train the RL agent")
    parser.add_argument("--config", type=str, default="td3_lss.yaml")
    parser.add_argument("--resume_last", action="store_true")
    parser.add_argument("--alg", type=str, default="td3")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        alg = TD3(env, config)
        if args.resume_last:
            alg.load_ckpt("checkpoints/rl/last_{}.ckpt".format(config["experiment_name"]))
        alg.train()


if __name__ == "__main__":
    main()
