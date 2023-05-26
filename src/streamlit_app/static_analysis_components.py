import plotly.express as px
import streamlit as st
import torch as t
from fancy_einsum import einsum
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX


from .utils import fancy_histogram, fancy_imshow


def show_qk_circuit(dt):
    with st.expander("show QK circuit"):
        st.write(
            """
            Usually the QK circuit uses the embedding twice but since we are interested in Atten to
            """
        )
        st.latex(
            r"""
            QK_{circuit} = W_{E(state)}^T W_Q^T W_K W_{E(RTG)}
            """
        )

        W_E_rtg = dt.reward_embedding[0].weight
        W_E_state = dt.state_embedding.weight
        W_Q = dt.transformer.blocks[0].attn.W_Q
        W_K = dt.transformer.blocks[0].attn.W_K

        W_QK = einsum(
            "head d_mod_Q d_head, head d_mod_K d_head -> head d_mod_Q d_mod_K",
            W_Q,
            W_K,
        )
        # st.write(W_QK.shape)

        # W_QK_full = W_E_rtg.T @ W_QK @ W_E_state
        W_QK_full = W_E_state.T @ W_QK @ W_E_rtg

        n_heads = dt.transformer_config.n_heads
        n_observations = dt.environment_config.observation_space.shape[0]
        W_QK_full_reshaped = W_QK_full.reshape(
            n_heads, 1, n_observations
        )

        selection_columns = st.columns(2)

        with selection_columns[0]:
            heads = st.multiselect(
                "Select Heads",
                options=list(range(n_heads)),
                key="head qk",
                default=[0],
            )

        format_func = format_function
        for head in heads:
            st.write("Head", head)
            fancy_imshow(
                        W_QK_full_reshaped[head].cpu().T.detach().numpy(),
                        color_continuous_midpoint=0,
                    )


def format_function(idx):
    return str(idx)

def show_ov_circuit(dt):
    with st.expander("Show OV Circuit"):
        st.subheader("OV circuits")

        st.latex(
            r"""
            OV_{circuit} = W_{U(action)} W_O W_V W_{E(State)}
            """
        )

        W_U = dt.action_predictor.weight
        W_O = dt.transformer.blocks[0].attn.W_O
        W_V = dt.transformer.blocks[0].attn.W_V
        W_E = dt.state_embedding.weight
        W_OV = W_V @ W_O

        # st.plotly_chart(px.imshow(W_OV.detach().numpy(), facet_col=0), use_container_width=True)
        OV_circuit_full = W_E.T @ W_OV @ W_U.T

        obs_shape = dt.environment_config.observation_space.shape
        n_actions = W_U.shape[0]
        n_heads = dt.transformer_config.n_heads
        OV_circuit_full_reshaped = OV_circuit_full.reshape(
            n_heads, obs_shape[0], n_actions
        )


        format_func = format_function



        selection_columns = st.columns(3)
        with selection_columns[0]:
            heads = st.multiselect(
                "Select Heads",
                options=list(range(n_heads)),
                key="head ov",
                default=[0],
            )

        with selection_columns[1]:
            selected_actions = st.multiselect(
                "Select Actions",
                options=list(range(n_actions)),
                key="actions ov",
                format_func=lambda x: format_function(x),#TODO change to action specific thing
                default=[0],
            )

        for head in heads:
            for action in selected_actions:
                st.write(f"Head {head} - {format_function(action)}")#TODO change to action specific thing
                

                fancy_imshow(
                            OV_circuit_full_reshaped[head,action]
                            .cpu().T.detach().numpy(),
                            color_continuous_midpoint=0,
                        )


def show_time_embeddings(dt, logit_dir):
    with st.expander("Show Time Embeddings"):
        if dt.time_embedding_type == "linear":
            time_steps = t.arange(100).unsqueeze(0).unsqueeze(-1).to(t.float32)
            time_embeddings = dt.get_time_embeddings(time_steps).squeeze(0)
        else:
            time_embeddings = dt.time_embedding.weight
        
        max_timestep = st.slider(
            "Max timestep",
            min_value=1,
            max_value=time_embeddings.shape[0] - 1,
            value=time_embeddings.shape[0] - 1,
        )
        time_embeddings = time_embeddings[: max_timestep + 1]
        dot_prod = time_embeddings @ logit_dir
        dot_prod = dot_prod.detach()

        show_initial = st.checkbox("Show initial time embedding", value=True)
        fig = px.line(dot_prod)
        fig.update_layout(
            title="Time Embedding Dot Product",
            xaxis_title="Time Step",
            yaxis_title="Dot Product",
            legend_title="",
        )
        # remove legend
        fig.update_layout(showlegend=False)
        if show_initial:
            fig.add_vline(
                x=st.session_state.timesteps[0][-1].item(),
                line_dash="dash",
                line_color="red",
                annotation_text="Current timestep",
            )
        st.plotly_chart(fig, use_container_width=True)

        def calc_cosine_similarity_matrix(matrix: t.Tensor) -> t.Tensor:
            # Check if the input matrix is square
            # assert matrix.shape[0] == matrix.shape[1], "The input matrix must be square."

            # Normalize the column vectors
            norms = t.norm(
                matrix, dim=0
            )  # Compute the norms of the column vectors
            normalized_matrix = (
                matrix / norms
            )  # Normalize the column vectors by dividing each element by the corresponding norm

            # Compute the cosine similarity matrix using matrix multiplication
            return t.matmul(normalized_matrix.t(), normalized_matrix)

        similarity_matrix = calc_cosine_similarity_matrix(time_embeddings.T)
        st.plotly_chart(px.imshow(similarity_matrix.detach().cpu().numpy()))


def show_rtg_embeddings(dt, logit_dir):
    with st.expander("Show RTG Embeddings"):
        batch_size = 1028
        if st.session_state.allow_extrapolation:
            min_value = -10
            max_value = 10
        else:
            min_value = -1
            max_value = 1
        rtg_range = st.slider(
            "RTG Range",
            min_value=min_value,
            max_value=max_value,
            value=(-1, 1),
            step=1,
        )

        min_rtg = rtg_range[0]
        max_rtg = rtg_range[1]

        rtg_range = t.linspace(min_rtg, max_rtg, 100).unsqueeze(-1)

        rtg_embeddings = dt.reward_embedding(rtg_range).squeeze(0)

        dot_prod = rtg_embeddings @ logit_dir
        dot_prod = dot_prod.detach()

        show_initial = st.checkbox("Show initial RTG embedding", value=True)

        fig = px.line(x=rtg_range.squeeze(1).detach().numpy(), y=dot_prod)
        fig.update_layout(
            title="RTG Embedding Dot Product",
            xaxis_title="RTG",
            yaxis_title="Dot Product",
            legend_title="",
        )
        # remove legend
        fig.update_layout(showlegend=False)
        if show_initial:
            fig.add_vline(
                x=st.session_state.rtg[0][0].item(),
                line_dash="dash",
                line_color="red",
                annotation_text="Initial RTG",
            )
        st.plotly_chart(fig, use_container_width=True)
