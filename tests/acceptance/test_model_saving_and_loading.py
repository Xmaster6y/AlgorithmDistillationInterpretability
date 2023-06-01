import json
import os

import pytest
import torch

from src.config import (
    EnvironmentConfig,
    OfflineTrainConfig,
    OnlineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)

from src.sar_transformer.runner import store_transformer_model
from src.sar_transformer.utils import load_algorithm_distillation_transformer, load_concat_transformer 
from src.models.trajectory_transformer import DecisionTransformer


@pytest.fixture()
def cleanup_test_results() -> None:
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    if os.path.exists("tmp/model_data.pt"):
        os.remove("tmp/model_data.pt")


@pytest.fixture()
def run_config() -> RunConfig:
    return RunConfig(
        exp_name="Test-PPO-Basic",
        seed=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        track=False,
        wandb_project_name="PPO-MiniGrid",
        wandb_entity=None,
    )


@pytest.fixture()
def environment_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        env_id="MiniGrid-Dynamic-Obstacles-8x8-v0",
        view_size=3,
        max_steps=300,
        one_hot_obs=True,
        fully_observed=False,
        render_mode="rgb_array",
        capture_video=True,
        video_dir="videos",
    )


@pytest.fixture()
def online_config() -> OnlineTrainConfig:
    return OnlineTrainConfig(
        hidden_size=64,
        total_timesteps=2000,
        learning_rate=0.00025,
        decay_lr=True,
        num_envs=30,
        num_steps=64,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=30,
        update_epochs=4,
        clip_coef=0.4,
        ent_coef=0.25,
        vf_coef=0.5,
        max_grad_norm=2,
        trajectory_path="trajectories/MiniGrid-DoorKey-8x8-trajectories.pkl",
    )


@pytest.fixture()
def transformer_config() -> TransformerModelConfig:
    return TransformerModelConfig(
        d_model=128,
        n_heads=4,
        d_mlp=256,
        n_layers=2,
        n_ctx=2,
        layer_norm=False,
        state_embedding_type="grid",
        time_embedding_type="embedding",
        seed=1,
        device="cpu",
    )


@pytest.fixture()
def offline_config() -> OfflineTrainConfig:
    return OfflineTrainConfig(
        trajectory_path="trajectories/MiniGrid-DoorKey-8x8-trajectories.pkl",
        batch_size=128,
        lr=0.0001,
        weight_decay=0.0,
        prob_go_from_end=0.0,
        device="cpu",
        track=False,
        train_epochs=100,
        test_epochs=10,
        test_frequency=10,
        eval_frequency=10,
        eval_episodes=10,
        model_type="decision_transformer",
        initial_rtg=[0.0, 1.0],
        eval_num_envs=8,
    )


def test_load_algorithm_distillation_transformer(
    transformer_config,
    offline_config,
    environment_config,
    cleanup_test_results,
):
    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
    )

    path = "tmp/model_data.pt"
    store_transformer_model(
        path=path,
        model=model,
        offline_config=offline_config,
    )

    new_model = load_algorithm_distillation_transformer(path)

    assert_state_dicts_are_equal(new_model.state_dict(), model.state_dict())

    assert new_model.transformer_config == transformer_config
    assert new_model.environment_config == environment_config


def test_load_concat_transformer(
    transformer_config,
    offline_config,
    environment_config,
    cleanup_test_results,
):
    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
    )

    path = "tmp/model_data.pt"
    store_transformer_model(
        path=path,
        model=model,
        offline_config=offline_config,
    )

    new_model = load_concat_transformer(path)

    assert_state_dicts_are_equal(new_model.state_dict(), model.state_dict())

    assert new_model.transformer_config == transformer_config
    assert new_model.environment_config == environment_config


def assert_state_dicts_are_equal(dict1, dict2):
    keys1 = sorted(dict1.keys())
    keys2 = sorted(dict2.keys())

    assert keys1 == keys2

    for key1, key2 in zip(keys1, keys2):
        assert dict1[key1].equal(dict2[key2])
