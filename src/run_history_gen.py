import argparse
import os
from .utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--env_id", type=str, required=True, choices=[
        "DarkRoom",
        "DarkKeyDoor",
        "ArmedBandit",
        "SimpleDarkRoom",
        "SimpleDarkKeyDoor"])
    parser.add_argument("--n_seeds", type=int, default=500)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=12_000)
    parser.add_argument("--n_states", type=int, default=12)
    parser.add_argument("--n_actions", type=int, default=2)
    parser.add_argument("--max_env_len", type=int, default=12)
    parser.add_argument("--n_rollouts", type=int, default=4)
    args = parser.parse_args()
    
    for env_seed in range(args.seed_start, args.seed_start+args.n_seeds):
        print(f"Training policy {env_seed - args.seed_start}")
        venv = create_vector_environment(
            env_id=args.env_id,
            n_envs=1, 
            n_states=args.n_states,
            n_actions=args.n_actions, 
            max_steps=args.max_env_len,
            seed=env_seed
        )
        file_path = os.path.join(args.path, f"{env_seed}")
        # We want to use PPO for DarkRoom and DarkKeyRoom, and UCB for MultiArmedBandits
        if args.env_id == "ArmedBandit":
            train_ucb(
                venv=venv,
                file_name=file_path,
                n_steps=args.n_steps,
            )
        else:
            train_policy(
                venv=venv,
                file_name=file_path,
                n_steps=args.n_steps,
                buffer_size=args.max_env_len * args.n_rollouts
            )
