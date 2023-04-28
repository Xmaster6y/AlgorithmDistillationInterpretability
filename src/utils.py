import dataclasses
import gzip
import json
import lzma
import os
import pickle
import time
from typing import Dict

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import PPO
from typeguard import typechecked
import argparse
import numpy as np
import os
import torch
import wandb

from generation import *
from config import ConfigJsonEncoder


class TrajectoryWriter:
    """
    The trajectory writer is responsible for writing trajectories to a file.
    During each rollout phase, it will collect:
        - the observations
        - the actions
        - the rewards
        - the dones
        - the infos
    And store them in a set of lists, indexed by batch b and time t.
    """

    def __init__(
        self,
        path,
        run_config,
        environment_config,
        online_config,
        model_config=None,
    ):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.truncated = []
        self.infos = []
        self.path = path

        args = (
            run_config.__dict__
            | environment_config.__dict__
            | online_config.__dict__
        )
        if model_config is not None:
            args = args | model_config.__dict__

        self.args = args

    @typechecked
    def accumulate_trajectory(
        self,
        next_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        truncated: np.ndarray,
        action: np.ndarray,
        info: Dict,
    ):
        self.observations.append(next_obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.truncated.append(truncated)
        self.infos.append(info)

    def tag_terminated_trajectories(self):
        """
        Tag the last trajectory in each batch as done.
        This is needed when an episode in a minibatch is ended because the
        timesteps limit has been reached but the episode may not have been truncated
        or ended in the environment.

        I don't love this solution, but it will do for now.
        """
        n_envs = len(self.dones[-1])
        for i in range(n_envs):
            self.truncated[-1][i] = True

    def write(self, upload_to_wandb: bool = False):
        data = {
            "observations": np.array(self.observations, dtype=np.float),
            "actions": np.array(self.actions, dtype=np.int64),
            "rewards": np.array(self.rewards, dtype=np.float),
            "dones": np.array(self.dones, dtype=bool),
            "truncated": np.array(self.truncated, dtype=bool),
            "infos": np.array(self.infos, dtype=object),
        }
        if dataclasses.is_dataclass(self.args):
            metadata = {
                # Args such as ppo args
                "args": json.dumps(self.args, cls=ConfigJsonEncoder),
                "time": time.time(),  # Time of writing
            }
        else:
            metadata = {
                "args": self.args,  # Args such as ppo args
                "time": time.time(),  # Time of writing
            }

        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

        # use lzma to compress the file
        if self.path.endswith(".xz"):
            print(f"Writing to {self.path}, using lzma compression")
            with lzma.open(self.path, "wb") as f:
                pickle.dump({"data": data, "metadata": metadata}, f)
        elif self.path.endswith(".gz"):
            print(f"Writing to {self.path}, using gzip compression")
            with gzip.open(self.path, "wb") as f:
                pickle.dump({"data": data, "metadata": metadata}, f)
        else:
            print(f"Writing to {self.path}")
            with open(self.path, "wb") as f:
                pickle.dump({"data": data, "metadata": metadata}, f)

        if upload_to_wandb:
            artifact = wandb.Artifact(
                self.path.split("/")[-1], type="trajectory"
            )
            artifact.add_file(self.path)
            wandb.log_artifact(artifact)

        print(f"Trajectory written to {self.path}")


def pad_tensor(
    tensor, length=100, ignore_first_dim=True, pad_token=0, pad_left=False
):
    if ignore_first_dim:
        if tensor.shape[1] < length:
            pad_shape = (
                tensor.shape[0],
                length - tensor.shape[1],
                *tensor.shape[2:],
            )
            pad = torch.ones(pad_shape) * pad_token

            if pad_left:
                tensor = torch.cat([pad, tensor], dim=1)
            else:
                tensor = torch.cat([tensor, pad], dim=1)

        return tensor
    else:
        if tensor.shape[0] < length:
            pad_shape = (length - tensor.shape[0], *tensor.shape[1:])
            pad = torch.ones(pad_shape) * pad_token

            if pad_left:
                tensor = torch.cat([pad, tensor], dim=0)
            else:
                tensor = torch.cat([tensor, pad], dim=0)

        return tensor


class DictList(dict):
    """A dictionary of lists of same size. Dictionary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, input_list):
        if isinstance(input_list, dict):
            super().__init__(input_list)
        elif isinstance(input_list, list):
            keys = input_list[0].keys()
            stacked_dict = {
                key: torch.stack([getattr(dl, key) for dl in input_list])
                for key in keys
            }
            super().__init__(stacked_dict)
        else:
            raise ValueError(
                "Input should be either a dictionary or a list of DictLists containing tensors."
            )

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value


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


def create_environment_from_id(env_id, n_states, n_actions, max_steps, seed):
    assert env_id in ["DarkKeyDoor", "DarkRoom", "ArmedBandit"]  # Only supports certain environments

    if env_id == "DarkKeyDoor":
        env = DarkKeyDoor(n_states, n_actions, max_steps, seed=seed)
    elif env_id == "DarkRoom":
        env = DarkRoom(n_states, n_actions, max_steps, seed=seed)
    elif env_id == "ArmedBandit":
        assert n_states == 1 and max_steps == 1, "--n_states and --max_env_len should be set to 1"  # Armed Bandit only supports single state
        env = MultiArmedBandit(n_actions, max_steps, seed=seed)

    return env


def create_vector_environment(env_id, n_envs, n_states, n_actions,  max_steps, seed):
    def create_env():
        # Create DarkKeyDoor environment with fixed seed
        env = create_environment_from_id(env_id, n_states, n_actions, max_steps, seed)
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


def train_ucb(venv, file_name, n_steps):
    # Train model using upper confidence bound algorithm
    env = venv.get_attr("env")[0]
    assert env.action_space.n < n_steps, "Not enough steps to run algorithm"
    # Initialize buffers
    means = np.zeros((env.action_space.n,))
    visits = np.zeros((env.action_space.n,))
    scores = np.zeros((env.action_space.n,))
    rollouts = {
        "observations": [],
        "actions"     : [],
        "rewards"     : [],
        "dones"       : []
    }
    # Pre-initialize tables
    for t in range(env.action_space.n):
        action = t
        obs, _ = env.reset()
        obs, reward, done, _, _ = env.step(action)
        # Add relevant to rollouts
        rollouts["observations"].append(obs)
        rollouts["actions"].append(action)
        rollouts["rewards"].append(reward)
        rollouts["dones"].append(done)
        # Update UCB scores
        means[action] = (means[action] * visits[action] + reward) / (visits[action] + 1)
        visits[action] += 1
    scores = means + np.sqrt(np.log(visits.sum()) / visits)
    # Run UCB
    for t in range(env.action_space.n, n_steps):
        action = np.argmax(scores)
        obs, _ = env.reset()
        obs, reward, done, _, _ = env.step(action)
        # Add relevant to rollouts
        rollouts["observations"].append(obs)
        rollouts["actions"].append(action)
        rollouts["rewards"].append(reward)
        rollouts["dones"].append(done)
        # Update UCB scores
        means[action] = (means[action] * visits[action] + reward) / (visits[action] + 1)
        visits[action] += 1
        scores = means + np.sqrt(np.log(t + 1) / visits)
    # Concatenate data to make everything a np array
    rollouts["observations"] = np.array(rollouts["observations"])
    rollouts["actions"] = np.array(rollouts["actions"])[:, None]
    rollouts["rewards"] = np.array(rollouts["rewards"])[:, None]
    rollouts["dones"] = np.array(rollouts["dones"])[:, None]
    # Write to a file
    np.savez(file_name, **rollouts)
