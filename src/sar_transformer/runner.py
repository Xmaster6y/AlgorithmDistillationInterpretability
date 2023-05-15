import json
import os
import time
import warnings
from typing import Callable

import torch as t
from src.generation import *
from src.sar_transformer.dataset import *
import wandb
from src.config import (
    ConfigJsonEncoder,
    EnvironmentConfig,
    OfflineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)
from src.models.trajectory_transformer import (
    CloneTransformer,
    ConcatTransformer,
    AlgorithmDistillationTransformer,
)

from .trainer import train
from .utils import get_max_len_from_model_type, store_transformer_model
from src.utils import create_environment_from_id



def run_ad_transformer(
    run_config: RunConfig,
    transformer_config: TransformerModelConfig,
    offline_config: OfflineTrainConfig,
    history_dataset: HistoryDataset,
    history_dataset_test: HistoryDataset,
    env_id: str,
):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    device = run_config.device
    max_len = get_max_len_from_model_type(  # TODO check its the correct one
        offline_config.model_type, transformer_config.n_ctx
    )

    train_loader = create_history_dataloader(
        history_dataset,
        offline_config.batch_size,
        256 * offline_config.batch_size
    )
    test_loader = create_history_dataloader(
        history_dataset_test,
        offline_config.batch_size,
        64 * offline_config.batch_size
    )

    env = create_environment_from_id(
        env_id,
        history_dataset.n_states,
        history_dataset.n_actions,
        history_dataset.episode_length,
        seed=100_000,
    )  # TODO maybe do something whith seed
    environment_config = EnvironmentConfig(
        env_id=f"Graph_{env_id}", env=env, device=device
    )

    if run_config.track:
        wandb_args = ({
            **run_config.__dict__, **transformer_config.__dict__, **offline_config.__dict__
        })
        run_name = f"{run_config.exp_name}__{run_config.seed}__{int(time.time())}"
        wandb.init(
            project=run_config.wandb_project_name,
            entity=run_config.wandb_entity,
            name=run_name,
            config=wandb_args,
        )
        """
        trajectory_visualizer = TrajectoryVisualizer(trajectory_data_set)#TODO fix that for graph env
        fig = trajectory_visualizer.plot_reward_over_time()
        wandb.log({"dataset/reward_over_time": wandb.Plotly(fig)})
        
        fig = trajectory_visualizer.plot_base_action_frequencies()
        wandb.log({"dataset/base_action_frequencies": wandb.Plotly(fig)})
        """
        wandb.log({"dataset/num_trajectories": history_dataset.n_histories})

    if offline_config.model_type == "algorithm_distillation":
        model = AlgorithmDistillationTransformer(
            environment_config=environment_config,
            transformer_config=transformer_config,
        )
    elif offline_config.model_type == "concat_transformer":
        model = ConcatTransformer(
            environment_config=environment_config,
            transformer_config=transformer_config,
        )
    else:
        raise Exception(f"Unknown model_type: {offline_config.model_type}")

    if run_config.track:
        wandb.watch(model, log="parameters")

    model = train(  # TODO look into batch size
        model,
        train_loader,
        test_loader,
        environment_config,
        lr=offline_config.lr,
        clip=1.0,
        device=offline_config.device,
        track=offline_config.track,
        train_epochs=offline_config.train_epochs,
        eval_frequency=offline_config.eval_frequency,
        eval_length=offline_config.eval_episodes,
        eval_temp=offline_config.eval_temp
    )

    if run_config.track:
        # save the model with pickle, then upload it
        # as an artifact, then delete it.
        # name it after the run name.
        if not os.path.exists("models"):
            os.mkdir("models")

        model_path = f"models/{run_name}.pt"

        store_transformer_model(
            path=model_path,
            model=model,
            offline_config=offline_config,
        )

        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        os.remove(model_path)
        wandb.finish()
