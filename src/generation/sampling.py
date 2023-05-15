from collections import defaultdict
from .utils import *
import numpy as np


def sample_transition_rules(n_states, n_actions, rng, use_mcmc=True):
    # Generates matrix of the form [n_states, n_actions, n_states]
    # Start by created a strongly-connected directed graph (a loop)
    T = np.zeros(
        (
            n_states,
            n_actions,
            n_states,
        )
    )
    adj_matrix = np.eye(n_states)
    adj_matrix = np.concatenate((adj_matrix[1:, :], adj_matrix[:1, :]), axis=0)
    T[:, 0, :] = adj_matrix
    # If there is a second action, set that equal to the edges reversed
    if n_actions >= 2:
        T[:, 1, :] = np.transpose(T[:, 0, :])
    # For the other actions, find a random state that we can go to
    for a in range(2, n_actions):
        for s in range(n_states):
            # Create a random edge
            sp = rng.integers(0, n_states)
            T[s, a, sp] = 1.0
    # Use MCMC to get a random SC digraph (hopefully close to a uniform sample)
    if use_mcmc:
        T = markov_chain_monte_carlo(T, rng)
    # Normalize the probabilities
    T = T / np.sum(T, axis=2)[:, :, None]
    return T


def markov_chain_monte_carlo(T, rng, n_its=1_000):
    # Performs Markov Chain Monte Carlo to sample a random strongly-connected graph
    n_states, n_actions, _ = T.shape
    for i in range(n_its):  # Not possible to calculate in general what the mixing time is here
        rand_prev_state = rng.integers(0, n_states)
        rand_next_state = rng.integers(0, n_states)
        rand_act = rng.integers(0, n_actions)
        # Zero out the current transition, and set it to be random
        curr_transition = np.argmax(T[rand_prev_state, rand_act, :])
        T[rand_prev_state, rand_act, curr_transition] = 0
        T[rand_prev_state, rand_act, rand_next_state] = 1
        # Check if still connected
        visited = np.zeros(n_states, dtype=bool)
        dfs(T.sum(1), rand_prev_state, visited)
        is_connected = visited[curr_transition]
        # If no longer connected, then revert
        if not is_connected:
            T[rand_prev_state, rand_act, curr_transition] = 1
            T[rand_prev_state, rand_act, rand_next_state] = 0
    return T


def sample_observation_rules(n_states):
    # How should we sample observations
    obs = np.eye(n_states)
    # Optionally, we can rotate the observations using a random orthonormal matrix to make it harder for the policies
    return obs


def sample_reward_rules(
    n_states,
    n_actions,
    n_rewards,
    rng,
    prob_use_old_state=1.0,
    prob_use_new_state=0.0,
    prob_use_action=0.33,
    prob_use_flag=0.33,
):
    # Rules are of the form (old_state, new_state, action, probability, value, flag)
    # Generate all possible rules
    rules = []
    examples = [
        (prev_state, new_state, action)
        for prev_state in range(-1, n_states)
        for new_state in range(-1, n_states)
        for action in range(-1, n_actions)
    ]
    # Sample from the list of valid rules until we have enough
    for nr in range(n_rewards):
        rule = [-1, -1, -1, 0, 0, -1]
        # Repeat until the rule is conditional on either the old state, new_state, or action
        idx = None
        while True:
            idx = rng.choice(np.arange(len(examples)))
            prev_state, new_state, action = examples[idx]
            # Accept-Reject sampling for old state
            if rng.random() < prob_use_old_state:
                if prev_state == -1:
                    continue
            else:
                if prev_state != -1:
                    continue
            # Accept-Reject sampling for new state
            if rng.random() < prob_use_new_state:
                if new_state == -1:
                    continue
            else:
                if new_state != -1:
                    continue
            # Accept-Reject sampling for action
            if rng.random() < prob_use_action:
                if action == -1:
                    continue
            else:
                if action != -1:
                    continue
            break
        rule[:3] = prev_state, new_state, action
        examples.pop(idx)
        # Decide whether or not to make reward conditional on hidden variable
        if rng.random() < prob_use_flag:
            rule[5] = 1.0
        # Reward is always 1, but probability of reward can change
        rule[3] = 1.0
        rule[4] = 1.0
        rules.append(rule)
    return rules


def sample_flag_rules(
    n_states,
    n_actions,
    n_flag_rules,
    rng,
    prob_use_old_state=1.0,
    prob_use_new_state=0.0,
    prob_use_action=0.33,
):
    # Rules are of the form (old_state, new_state, action, new flag value)
    flagrules = []
    for nr in range(n_flag_rules):
        flagrule = [-1, -1, -1, 1.0]
        while flagrule[0] == -1 and flagrule[1] == -1 and flagrule[2] == -1:
            if rng.random() < prob_use_old_state:
                flagrule[0] = rng.integers(0, n_states)
                # Check if there exists a sta
            if rng.random() < prob_use_new_state:
                flagrule[1] = rng.integers(0, n_states)
            if rng.random() < prob_use_action:
                flagrule[2] = rng.integers(0, n_actions)
        flagrule[3] = 1  # np.random.choice([1, 1, 1, 0])
        flagrules.append(flagrule)
    return flagrules
