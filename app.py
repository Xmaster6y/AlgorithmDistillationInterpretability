import time

import streamlit as st
import plotly.express as px

from src.streamlit_app.causal_analysis_components import show_ablation
from src.streamlit_app.components import (
    hyperpar_side_bar,
    record_keypresses,
    render_game_screen,
    render_trajectory_details,
    reset_button,
    reset_env_dt,
)
from src.streamlit_app.content import (
    analysis_help,
    help_page,

)
from src.streamlit_app.dynamic_analysis_components import (
    render_observation_view,
    show_attention_pattern,
    show_residual_stream_contributions_single,
)
from src.streamlit_app.setup import initialize_playground
from src.streamlit_app.static_analysis_components import (
    show_ov_circuit,
    show_qk_circuit,
    show_rtg_embeddings,
    show_time_embeddings,
)
from src.streamlit_app.visualizations import action_string_to_id

from src.streamlit_app.model_index import model_index

start = time.time()

st.set_page_config(
    page_title="Algorithm Distillation Interpretability",
    page_icon="assets/logofiles/Logo_black.ico",
)


with st.sidebar:
    
    st.image(
        "assets/logofiles/Logo_transparent.png", use_column_width="always"
    )
    st.title("Decision Transformer Interpretability")

    model_directory = "models"

    with st.form("model_selector"):
        selected_model_path = st.selectbox(
            label="Select Model",
            options=model_index.keys(),
            format_func=lambda x: model_index[x],
            key="model_selector",
        )
        if submitted := st.form_submit_button("Load Model"):
            reset_env_dt()

hyperpar_side_bar()




# st.session_state.max_len = 1
env, dt = initialize_playground(selected_model_path)
x, cache, tokens = render_game_screen(dt, env)

action_options = [f"Action {i}" for i in range(1, env.action_space.n + 1)]#TODO maybe needs a done option
action_string_to_id = {element: index for index, element in enumerate(action_options)}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}
record_keypresses()


with st.sidebar:
    st.subheader("Directional Analysis")
    if comparing := st.checkbox("comparing directions", value=True):
        positive_action_direction = st.selectbox(
            "Positive Action Direction",
            action_options,
            index=0,
        )
        negative_action_direction = st.selectbox(
            "Negative Action Direction",
            action_options,
            index=1,
        )
        positive_action_direction = action_string_to_id[
            positive_action_direction
        ]
        negative_action_direction = action_string_to_id[
            negative_action_direction
        ]

        logit_dir = (
            dt.action_predictor.weight[positive_action_direction]
            - dt.action_predictor.weight[negative_action_direction]
        )
    else:
        st.warning("Single Logit Analysis may be misleading.")
        selected_action_direction = st.selectbox(
            "Selected Action Direction",
            action_options,
            index=2,
        )
        selected_action_direction = action_string_to_id[
            selected_action_direction
        ]
        logit_dir = dt.action_predictor.weight[selected_action_direction]

    st.subheader("Analysis Selection")
    static_analyses = st.multiselect(
        "Select Static Analyses",
        ["Reward Embeddings", "Time Embeddings", "OV Circuit", "QK Circuit"],
    )
    dynamic_analyses = st.multiselect(
        "Select Dynamic Analyses",
        [
            "Residual Stream Contributions",
            "Attention Pattern",
            "Observation View",
        ],
    )
    causal_analyses = st.multiselect("Select Causal Analyses", ["Ablation"])
analyses = dynamic_analyses + static_analyses + causal_analyses

with st.sidebar:
    render_trajectory_details()
    reset_button()

if len(analyses) == 0:
    st.warning("Please select at least one analysis.")

if "reward Embeddings" in analyses:
    show_rtg_embeddings(dt, logit_dir)
if "Time Embeddings" in analyses:
    show_time_embeddings(dt, logit_dir)
if "QK Circuit" in analyses:
    show_qk_circuit(dt)
if "OV Circuit" in analyses:
    show_ov_circuit(dt)

if "Ablation" in analyses:
    show_ablation(dt, logit_dir=logit_dir, original_cache=cache)


if "Residual Stream Contributions" in analyses:
    show_residual_stream_contributions_single(dt, cache, logit_dir=logit_dir)
if "Attention Pattern" in analyses:
    show_attention_pattern(dt, cache)
if "Observation View" in analyses:
    render_observation_view(dt, tokens, logit_dir)


st.markdown("""---""")

with st.expander("Show history"):
    rendered_obss = st.session_state.rendered_obs
    trajectory_length = rendered_obss.shape[0]

    historic_actions = st.session_state.a[0, -trajectory_length:].flatten()

    if trajectory_length > 1:
        right_adjustment = 1 + st.session_state.max_len - trajectory_length

        state_number = st.slider(
            "State Number",
            min_value=right_adjustment,
            max_value=right_adjustment + trajectory_length - 1,
            step=1,
            format="State Number: %d",
        )

        i = state_number - right_adjustment
        action_name_func = (
            lambda a: "None" if a == 7 else action_id_to_string[a]
        )
        st.write(f"A{i}:", action_name_func(historic_actions[i].item()))
        st.write(f"A{i+1}:", action_name_func(historic_actions[i + 1].item()))
        st.plotly_chart(px.imshow(rendered_obss[i, :, :, :]))
    else:
        st.warning("No history to show")

st.markdown("""---""")

st.session_state.env = env
st.session_state.dt = dt

with st.sidebar:
    end = time.time()
    st.write(f"Time taken: {end - start}")

record_keypresses()

help_page()
analysis_help()
