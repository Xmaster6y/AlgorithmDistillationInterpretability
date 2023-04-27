from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
import numpy as np
from generation import *
import argparse
import os


class RecordHistory(BaseCallback):

    def __init__(self, file_name):
        super().__init__()
        self.rollouts = {
            "observations": [],
            "actions"     : [],
            "rewards"     : [],
            "dones"       : []
        }
        self.file_name = file_name

    def _on_step(self):
        # Retreive relevant information from locals
        obs = self.locals["obs_tensor"].cpu().numpy()[0]
        act = self.locals["actions"]
        rew = self.locals["rewards"]
        dones = self.locals["dones"]  # dones = done or truncated

        # Add to dictionary
        self.rollouts["observations"].append(obs)
        self.rollouts["actions"].append(act)
        self.rollouts["rewards"].append(rew)
        self.rollouts["dones"].append(dones)

    def _on_training_end(self):
        # Concatenate data to make everything a np array
        self.rollouts["observations"] = np.array(self.rollouts["observations"])
        self.rollouts["actions"] = np.array(self.rollouts["actions"])
        self.rollouts["rewards"] = np.array(self.rollouts["rewards"])
        self.rollouts["dones"] = np.array(self.rollouts["dones"])
        # Write to a file
        np.savez(self.file_name, **self.rollouts)


def create_vector_environment(seed, n_states, n_actions, n_envs, env_len):
    def create_env():
        # Create DarkKeyDoor environment with fixed seed
        env = DarkKeyDoor(n_states, n_actions, env_len, seed=seed)
        env = Monitor(env)
        return env
    # Add in a couple other things to assist in training
    venv = DummyVecEnv([create_env for _ in range(n_envs)])
    return venv


def train_policy(venv, file_name, n_steps, buffer_size):
    # Trains a recurrent policy to play env with seed
    # Writes the data to file_name
    model = RecurrentPPO("MlpLstmPolicy", venv, ent_coef=0.01, verbose=1, n_steps=buffer_size, batch_size=buffer_size)
    record_history = RecordHistory(file_name)
    model = model.learn(total_timesteps=n_steps, callback=record_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
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
        venv = create_vector_environment(env_seed, args.n_states, args.n_actions, 1, args.max_env_len)
        file_path = os.path.join(args.path, f"{env_seed}")
        train_policy(venv, file_path, args.n_steps, args.max_env_len * args.n_rollouts)
