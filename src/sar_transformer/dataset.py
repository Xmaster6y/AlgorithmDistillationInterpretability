import glob
import os

import torch
import numpy as np
import matplotlib.pyplot as plt


class HistoryDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 history_dir,
                 n_episodes_per_seq=10):

        self.history_dir = history_dir
        self.n_episodes_per_seq = n_episodes_per_seq
        self.load_histories()
        self.upper_bound = (self.n_episodes - self.n_episodes_per_seq) * self.episode_length + 1
        self.size = self.n_histories * self.upper_bound
        print(self.n_histories)

    def load_histories(self,):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

        files = list(glob.glob(os.path.join(self.history_dir, "*.npz")))
        data = np.load(files[0], allow_pickle=True)
        env = data["env"].item()
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.episode_length = env.max_steps

        for file_path in files:
            data = np.load(file_path)
            self.states.append(torch.from_numpy(data["observations"]))
            self.actions.append(torch.from_numpy(data["actions"]))
            self.rewards.append(torch.from_numpy(data["rewards"]))
            self.dones.append(torch.from_numpy(data["dones"]))

        self.states = torch.stack(self.states, dim=0)
        self.actions = torch.stack(self.actions, dim=0)
        self.rewards = torch.stack(self.rewards, dim=0)
        self.dones = torch.stack(self.dones, dim=0)

        self.n_histories = self.states.shape[0]
        self.n_episodes = self.states.shape[1] // self.episode_length

        self.timesteps = torch.arange(self.episode_length)[None, :, None].repeat(
            self.n_histories, self.n_episodes, 1)

        # Make sure the history boundaries are in the correct space
        assert torch.all(self.dones[:, (self.episode_length - 1)::self.episode_length, :])

    def __len__(self,):
        return self.size

    def __getitem__(self, idx):
        ep_idx = idx // (self.upper_bound)
        start_pos = idx % self.upper_bound
        end_pos = start_pos + self.episode_length * self.n_episodes_per_seq
        return (
            self.states[ep_idx, start_pos:end_pos],
            self.actions[ep_idx, start_pos:end_pos],
            self.rewards[ep_idx, start_pos:end_pos],
            self.timesteps[ep_idx, start_pos:end_pos]
        )


def create_history_dataloader(dataset, batch_size, n_samples):
    # Function to group together sampled examples
    def collate(data):
        states, actions, rewards, timesteps = zip(*data)
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        timesteps = torch.stack(timesteps, dim=0)
        return states, actions, rewards, timesteps
    # Create dataloader
    sampler = torch.utils.data.RandomSampler(dataset,
                                             replacement=True,
                                             num_samples=n_samples)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       sampler=sampler,
                                       collate_fn=collate)


def visualize_reward_over_hist(dataset, hist_idx, ep_len=12, rollout_len=48):
    # Visualize all learning histories
    rewards = dataset.rewards[hist_idx]
    n_steps, _ = rewards.shape
    returns = rewards.reshape(n_steps // rollout_len, rollout_len).sum(dim=-1) / (rollout_len // ep_len)
    plt.plot(returns)
    plt.show()
