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
        # create rng
        if seed is None:
            seed = np.random.randint(0, 2000)
        self.rng = np.random.default_rng(seed=seed)
        # Sample environment properties
        self.transition = sample_transition_rules(n_states, n_actions, self.rng, use_mcmc=use_mcmc)
        self.observations = sample_observation_rules(n_states)
        self.reward_rules = sample_reward_rules(
            n_states,
            n_actions,
            n_rewards,
            self.rng,
            prob_use_action=prob_use_action,
            prob_use_flag=prob_use_flag,
        )
        # Sample flag rules such that they have no overlap with the reward_rules
        # There must be no two flag and reward rules that share the same old state
        while True:
            self.flag_rules = sample_flag_rules(
                n_states, n_actions, n_flags, self.rng, prob_use_action=prob_use_action
            )
            # Resample flag rules if there is a flag rule with the same old state as a reward rule
            repeat = False
            for reward_rule in self.reward_rules:
                for flag_rule in self.flag_rules:
                    if reward_rule[0] == flag_rule[0]:
                        repeat = True
            if not repeat:
                break
        # Set environment options
        self.observation_space = gym.spaces.Box(0, 1.0, (n_states,))
        self.action_space = gym.spaces.Discrete(n_actions)
        # Set environment render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return self.observations[self.current_state]

    @property
    def steps_remaining(self):
        return self.max_steps - self.current_step

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_state = np.random.randint(
            0, self.n_states
        )  # Choose new starting position
        self.current_flag = 0  # Reset flag
        self.current_step = 0
        self.prev_state = None
        self.prev_action = None

        return self._get_obs(), {}

    def step(self, action):
        # Save previous state/action pair
        self.prev_state = self.current_state
        self.prev_action = action
        # Sample the next state from the transition matrix
        t_probs = self.transition[self.current_state, action]
        next_state = np.random.choice(self.n_states, 1, p=t_probs)[
            0
        ]  # Sample s' from categorical dist T[s, a]
        # Check to update flag
        for flag_old_state, flag_next_state, flag_action, flag_value in self.flag_rules[
            ::-1
        ]:
            # Check if the rule meets the desired criteria
            if flag_old_state == -1 or flag_old_state == self.current_state:
                if flag_next_state == -1 or flag_next_state == next_state:
                    if flag_action == -1 or flag_action == action:
                        # Set the new flag value and break
                        # Newer rules override older ones so we break when we meet the first rule that applies
                        self.current_flag = flag_value
                        break
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
            if reward_old_state == -1 or reward_old_state == self.current_state:
                if reward_next_state == -1 or reward_next_state == next_state:
                    if reward_action == -1 or reward_action == action:
                        if reward_flag == -1 or reward_flag == self.current_flag:
                            if np.random.random() < reward_prob:
                                reward = reward_value
        # Calculate remaining timesteps and if done
        self.current_state = next_state
        self.current_step += 1
        done = self.current_step == self.max_steps
        obs = self._get_obs()
        # returns obs, reward, terminated, truncated, info
        return obs, reward, done, False, {}

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
        edge_colors = []
        for ls in range(self.transition.shape[0]):
            for la in range(self.transition.shape[1]):
                for ns in range(self.transition.shape[2]):
                    if self.transition[ls, la, ns] > 0:
                        if ls == self.prev_state and la == self.prev_action:
                            edge_colors.append("darkgoldenrod")
                        else:
                            edge_colors.append("black")
                        graph.add_edge(ls, ns)
        # Label current state, reward states, and flag states
        for reward_rule in self.reward_rules:
            reward_node = reward_rule[0]
            if reward_node != -1:
                node_color_map[reward_node] = "lightgreen"
        for flag_rule in self.flag_rules:
            flag_node = flag_rule[0]
            if flag_node != -1:
                node_color_map[flag_node] = "salmon"
        node_edge_sizes[self.current_state] = 5
        # Draw image and return image array
        fig, ax = plt.subplots(figsize=(6, 6))
        nx.draw_circular(
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
        # We need to reinit probs for the rewards
        for reward_rule in self.reward_rules:
            reward_rule[3] = self.rng.beta(1, 5)
        self.reward_rules[
            self.rng.integers(0, len(self.reward_rules))
            ][3] = 1.0  # Make one of the arms the best

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
    Random General Task
    """

    def __init__(self, n_states, n_actions, n_steps, seed=None):
        # 50% Dark Room, 50% Dark Key Door
        super().__init__(
            n_states, n_actions, 1, 1, n_steps, prob_use_flag=0.5, seed=seed
        )
