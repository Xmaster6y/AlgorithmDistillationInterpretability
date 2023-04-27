import glob
import os

import torch
import numpy as np
import matplotlib.pyplot as plt


class HistoryDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 history_dir,
                 episode_length=128,
                 max_episodes=100,
                 n_episodes_per_seq=2):

        self.history_dir = history_dir
        self.episode_length = episode_length
        self.max_episodes = max_episodes
        self.n_episodes_per_seq = n_episodes_per_seq
        
        # Load the data and calculate statistics
        self.load_histories()
        self.upper_bound = max_episodes - n_episodes_per_seq + 1
        self.size = self.upper_bound * self.states.shape[0]
    
    def load_histories(self,):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        cutoff = self.episode_length * self.max_episodes
        for file_path in glob.glob(os.path.join(self.history_dir, "*.npz")):
            data = dict(np.load(file_path))
            assert data["observations"].shape[0] > cutoff  # Make sure there is enough data in each history
            self.states.append(torch.from_numpy(data["observations"])[:cutoff])
            self.actions.append(torch.from_numpy(data["actions"])[:cutoff])
            self.rewards.append(torch.from_numpy(data["rewards"])[:cutoff])
            self.dones.append(torch.from_numpy(data["dones"])[:cutoff])

        self.states = torch.stack(self.states, dim=0)
        self.actions = torch.stack(self.actions, dim=0)
        self.rewards = torch.stack(self.rewards, dim=0)
        self.dones = torch.stack(self.dones, dim=0)
        
        # Make sure the history boundaries are in the correct space
        assert torch.all(self.dones[:, (self.episode_length - 1)::self.episode_length, :])

    def __len__(self,):
        return self.size

    def __getitem__(self, idx):
        ep_idx = idx // self.upper_bound
        start_pos = (idx % self.upper_bound) * self.episode_length
        end_pos = start_pos + self.episode_length * self.n_episodes_per_seq
        return (
            self.states[ep_idx, start_pos:end_pos],
            self.actions[ep_idx, start_pos:end_pos],
            self.rewards[ep_idx, start_pos:end_pos]
        )


def create_history_dataloader(dataset, batch_size):
    # Function to group together sampled examples
    def collate(data):
        states, actions, rewards = zip(*data)
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        return states, actions, rewards
    # Create dataloader
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       collate_fn=collate)


def visualize_reward_over_hist(dataset, hist_idx, ep_len=128):
    # Visualize all learning histories
    rewards = dataset.rewards[hist_idx]
    n_steps, _ = rewards.shape
    returns = rewards.reshape(n_steps // ep_len, ep_len).sum(dim=-1) / 8
    plt.plot(returns)
    plt.show()
