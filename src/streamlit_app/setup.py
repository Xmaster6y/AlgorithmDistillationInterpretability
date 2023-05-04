import streamlit as st
import torch as t
import numpy as np

from src.sar_transformer.utils import (
    get_max_len_from_model_type,
)
from .environment import get_env_and_dt, get_action_from_user

action_string_to_id = {#TODO change the
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6,
}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}

def initialize_playground(model_path, initial_rtg):
    if "env" not in st.session_state or "dt" not in st.session_state:
        env, dt = get_env_and_dt(model_path)
        

        max_len = get_max_len_from_model_type(
            dt.model_type, dt.transformer_config.n_ctx
        )
        n_obs = env.observation_space.shape[0]
        obs = np.zeros((1, max_len, n_obs))
        initial_obs, _ = env.reset()
        obs[0, 0] = initial_obs
        actions = np.zeros((1, max_len, 1))
        reward = np.zeros((1, max_len, 1))
        timesteps = np.zeros((1, max_len, 1))
        current_episode = 0

        rendered_obs = t.from_numpy(env.render()).unsqueeze(0)

        st.session_state.max_len = max_len
        st.session_state.obs = obs
        st.session_state.rendered_obs = rendered_obs
        st.session_state.reward = reward
        st.session_state.a = actions
        st.session_state.timesteps = timesteps
        st.session_state.dt = dt

    else:
        env = st.session_state.env
        dt = st.session_state.dt

    if "action" in st.session_state:
        action = st.session_state.action
        if isinstance(action, str):
            action = action_string_to_id[action]
        st.write(
            f"just took action '{action_id_to_string[st.session_state.action]}'"
        )
        del action
        del st.session_state.action
    else:
        get_action_from_user(env, initial_rtg)

    return env, dt
