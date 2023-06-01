import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from .utils import *
from .sampling import *
from dataclasses import dataclass

@dataclass
class GeneralTask(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        n_states,
        n_actions,
        n_rewards,
        n_flags,
        max_steps,
        render_mode="rgb_array",
        seed=None,
        use_mcmc=True,
        prob_use_action=0.33,
        prob_use_flag=0.33,
    ):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_rewards = n_rewards
        self.n_flags = n_flags
        self.max_steps = max_steps
        self.use_mcmc = use_mcmc
        self.prob_use_action = prob_use_action
        self.prob_use_flag = prob_use_flag
        if seed is None:
            seed = self.np_random.integers(0, 2000)
        #  We keep a separate rng from self.np_random that is only used in self.generate() 
        self.rng = np.random.default_rng(seed=seed)
        self.generate()
        # Set environment options
        self.observation_space = gym.spaces.Box(0, 1.0, (n_states,))
        self.action_space = gym.spaces.Discrete(n_actions)
        # Set environment render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def generate(self):
        # Sample environment properties
        self.transition = sample_transition_rules(
            self.n_states, self.n_actions, self.rng, use_mcmc=self.use_mcmc)
        self.observations = sample_observation_rules(self.n_states)
        self.reward_rules = sample_reward_rules(
            self.n_states,
            self.n_actions,
            self.n_rewards,
            self.rng,
            prob_use_action=self.prob_use_action,
            prob_use_flag=self.prob_use_flag,
        )
        # Sample flag rules such that they have no overlap with the reward_rules
        # There must be no two flag and reward rules that share the same old state
        while True:
            self.flag_rules = sample_flag_rules(
                self.n_states,
                self.n_actions,
                self.n_flags,
                self.rng,
                prob_use_action=self.prob_use_action
            )
            # Resample flag rules if there is a flag rule with the same old state as a reward rule
            repeat = False
            for reward_rule in self.reward_rules:
                for flag_rule in self.flag_rules:
                    if reward_rule[0] == flag_rule[0]:
                        repeat = True
            if not repeat:
                break

    def set_seed(self,selected_seed=None):
        self.rng = np.random.default_rng(seed=selected_seed)#if none its random

    def _get_obs(self):
        return self.observations[self.current_state]
    

    @property
    def steps_remaining(self):
        return self.max_steps - self.current_step

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_state = self.np_random.integers(
            0, self.n_states
        )  # Choose new starting position
        self.current_flag = 0  # Reset flag
        self.current_step = 0
        self.prev_state = self.current_state
        self.prev_action = None

        return self._get_obs(), {}

    def step(self, action):
        # Save previous state/action pair
        self.prev_state = self.current_state
        self.prev_action = action
        # Sample the next state from the transition matrix
        t_probs = self.transition[self.current_state, action]
        next_state = self.np_random.choice(self.n_states, 1, p=t_probs)[
            0
        ]  # Sample s' from categorical dist T[s, a]
        # Calculate rewards
        reward = 0
        for (
            reward_old_state,
            reward_next_state,
            reward_action,
            reward_prob,
            reward_value,
            reward_flag,
        ) in self.reward_rules[::-1]:
            if reward_next_state in [-1, next_state]:
                if reward_action in [-1, action]:
                    if reward_old_state == -1 or reward_old_state == self.current_state:
                        if reward_flag == -1 or reward_flag == self.current_flag:
                            if self.np_random.random() <= reward_prob:
                                reward = reward_value
                            break
        # Calculate remaining timesteps and if done
        self.current_state = next_state
        self.update_flag()
        self.current_step += 1
        done = self.current_step == self.max_steps
        obs = self._get_obs()
        # returns obs, reward, terminated, truncated, info
        return obs, reward, done, False, {}

    def update_flag(self):
        # Check to update flag
        for flag_old_state, flag_next_state, flag_action, flag_value in self.flag_rules[::-1]:
            if flag_old_state in [-1, self.prev_state]:
                if flag_next_state in [-1, self.current_state]:
                    if flag_action in [-1, self.prev_action]:
                        # Set the new flag value and break
                        # Newer rules override older ones so we break when we meet the first rule that applies
                        self.current_flag = flag_value
                        break

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        graph = nx.DiGraph()
        # Add nodes to graph
        node_color_map = []
        node_edge_sizes = []
        for s in range(self.n_states):
            graph.add_node(s)
            node_color_map.append("lightblue")
            node_edge_sizes.append(0)
        # Add edges to graph
        for ls in range(self.transition.shape[0]):
            for la in range(self.transition.shape[1]):
                for ns in range(self.transition.shape[2]):
                    if self.transition[ls, la, ns] > 0:
                        if ls == self.prev_state and la == self.prev_action:
                            color = "darkgoldenrod"
                        else:
                            color = "black"
                        graph.add_edge(ls, ns, color=color)
        edge_colors = [graph[u][v]['color'] for u, v in graph.edges()]
        # Label current state, reward states, and flag states
        for reward_rule in self.reward_rules:
            reward_node = reward_rule[0]
            if reward_node != -1:
                node_color_map[reward_node] = "lightgreen"
        for flag_rule in self.flag_rules:
            flag_node = flag_rule[0]
            if flag_node != -1 and self.current_flag == 0:
                node_color_map[flag_node] = "salmon"
        node_edge_sizes[self.current_state] = 5
        # Draw image and return image array
        fig, ax = plt.subplots(figsize=(6, 6))
        nx.draw_kamada_kawai(
            graph,
            ax=ax,
            connectionstyle='arc3, rad = 0.1',
            with_labels=True,
            node_color=node_color_map,
            font_weight="bold",
            edgecolors="darkgoldenrod",
            linewidths=node_edge_sizes,
            edge_color=edge_colors,
            arrowsize=15,
            width=1.5,
            node_size=800,
        )
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        output = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return output

@dataclass
class MultiArmedBandit(GeneralTask):
    """
    Environment with n_arms actions, each with a different probability of
    yielding reward.  Probabilities are sampled from Beta distribution.
    """

    def __init__(self, n_arms, seed=None):
        super().__init__(
            1,
            n_arms,
            n_arms,
            0,
            1,
            prob_use_action=1.0,
            prob_use_flag=0.0,
            seed=seed,
            use_mcmc=False
        )

    def generate(self):
        super().generate()
        # We need to reinit probs for the rewards
        rew_probs = np.linspace(0, 1, self.n_actions)
        rew_probs = self.rng.permutation(rew_probs)
        for idx, reward_rule in enumerate(self.reward_rules):
            reward_rule[3] = rew_probs[idx]

@dataclass
class NavigationBandit(GeneralTask):
    """
    Each state has an action to pull a lever that will yield reward with some
    probability, or another action that puts the agent at a random state.
    """

    def __init__(self, n_states, n_steps, seed=None):
        super().__init__(
            n_states,
            2,
            n_states,
            0,
            n_steps,
            prob_use_action=0.0,
            prob_use_flag=0.0,
            seed=seed,
            use_mcmc=False
        )

    def generate(self):
        super().generate()
        # We need to reinit probs for the rewards
        for reward_rule in self.reward_rules:
            reward_rule[3] = self.rng.random()
        # We need to re-init transition
        self.transition[:, 0, :] = np.eye(n_states)
        self.transition[:, 1, :] = 1
        self.transition = self.transition / np.sum(self.transition, axis=2)[:, :, None]

@dataclass
class DarkRoom(GeneralTask):
    """
    DiGraph with randomly sampled transitions, and a single reward node
    """

    def __init__(self, n_states, n_actions, n_steps, seed=None):
        # Never has a flag in the reward rule
        super().__init__(
            n_states, n_actions, 1, 0, n_steps, prob_use_flag=0.0, seed=seed
        )

@dataclass
class DarkKeyDoor(GeneralTask):
    """
    DiGraph with randomly sampled transitions, and a single reward node that
    only gets activated after you visit a flag node
    """

    def __init__(self, n_states, n_actions, n_steps, seed=None):
        # Always has a flag in the reward rule
        super().__init__(
            n_states, n_actions, 1, 1, n_steps, prob_use_flag=1.0, seed=seed
        )

@dataclass
class SimpleDarkRoom(GeneralTask):
    """
    Graph with a single reward node
    """

    def __init__(self, n_states, n_actions, n_steps, seed=None):
        # Never has a flag in the reward rule
        super().__init__(
            n_states, n_actions, 1, 0, n_steps, prob_use_flag=0.0, seed=seed, use_mcmc=False
        )


class SimpleDarkKeyDoor(GeneralTask):
    """
    Graph with a single reward node that only gets activated after you visit a
    flag node
    """

    def __init__(self, n_states, n_actions, n_steps, seed=None):
        # Always has a flag in the reward rule
        super().__init__(
            n_states, n_actions, 1, 1, n_steps, prob_use_flag=1.0, seed=seed, use_mcmc=False
        )

@dataclass
class RandomTask(GeneralTask):
    """
    50% Chance of DarkRoom, 50% Chance of DarkKeyDoor
    """

    def __init__(self, n_states, n_actions, n_steps, seed=None):
        # 50% Dark Room, 50% Dark Key Door
        super().__init__(
            n_states, n_actions, 1, 1, n_steps, prob_use_flag=0.5, seed=seed
        )


class MetaRLTask(GeneralTask):
    """
    Repeats the same task n_task times
    """

    def __init__(
        self,
        n_states,
        n_actions,
        n_rewards,
        n_flags,
        max_steps,
        n_trials,
        render_mode="rgb_array",
        seed=None,
        use_mcmc=True,
        prob_use_action=0.33,
        prob_use_flag=0.33,
    ):
        super().__init__(
            n_states,
            n_actions,
            n_rewards,
            n_flags,
            max_steps,
            render_mode,
            seed,
            use_mcmc,
            prob_use_action,
            prob_use_flag,
        )

        self.observation_space = gym.spaces.Box(0, 1.0, (n_states+n_actions+2,))
        self.n_trials = n_trials

    @property
    def steps_remaining(self):
        return self.n_trials * self.max_steps - self.current_step

    def _get_obs(self):
        extra_space = np.zeros(self.n_actions + 2)
        return np.concatenate((
            self.observations[self.current_state],
            extra_space
        ), axis=0)

    def reset(self, *args, **kwargs):
        self.generate()
        self.current_trial = 0
        return super().reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            self.current_trial += 1
            obs, _ = super().reset()
            obs[-1] = 1.0  # Set done
        terminated = self.current_trial == self.n_trials
        # Observations includes previous reward and action
        obs[self.n_states + action] = 1.0
        obs[-2] = reward
        return obs, reward, terminated, truncated, info


class MetaRLArmedBandit(MetaRLTask):
    """
    Environment with n_arms actions, each with a different probability of
    yielding reward.  Probabilities are sampled from Beta distribution.
    """

    def __init__(self, n_arms, n_trials, seed=None,):
        super().__init__(
            n_states=1,
            n_actions=n_arms,
            n_rewards=n_arms,
            n_flags=0,
            max_steps=1,
            n_trials=n_trials,
            render_mode="rgb_array",
            seed=seed,
            use_mcmc=False,
            prob_use_action=1.0,
            prob_use_flag=0.0,
        )

    def generate(self):
        super().generate()
        # We need to reinit probs for the rewards
        for reward_rule in self.reward_rules:
            reward_rule[3] = self.rng.beta(1, 5)
            reward_rule[4] = 0.05
        self.reward_rules[
            self.rng.integers(0, len(self.reward_rules))
            ][3] = 1.0  # Make one of the arms the best
