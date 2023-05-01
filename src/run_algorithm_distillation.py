"""
This file is the entry point for running the decision transformer.
"""
import torch as t

from .config import OfflineTrainConfig, RunConfig, TransformerModelConfig
from .sar_transformer.runner import run_decision_transformer
from .sar_transformer.utils import parse_args
from .environments.environments import make_env
from src.sar_transformer.dataset import *

if __name__ == "__main__":
    args = parse_args()

    run_config = RunConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        device="cuda" if args.cuda and t.cuda.is_available() else "cpu",
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,

    )

    TIME_EMBEDDING_TYPE = (
        "linear" if args.linear_time_embedding else "embedding"
    )

    offline_config = OfflineTrainConfig(
        model_type=args.model_type,
        trajectory_path=args.trajectory_path,
        pct_traj=args.pct_traj,
        train_epochs=args.train_epochs,
        test_epochs=args.test_epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        test_frequency=args.test_frequency,
        eval_episodes=args.eval_episodes,
        eval_num_envs=args.eval_num_envs,
        prob_go_from_end=args.prob_go_from_end,
        eval_max_time_steps=args.eval_max_time_steps,
        track=args.track,
        device=run_config.device
    )
    history_dataset = HistoryDataset(offline_config.trajectory_path)
    context_len = history_dataset.n_episodes_per_seq * history_dataset.episode_length * 3 - 2

    transformer_model_config = TransformerModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        layer_norm=args.layer_norm,
        time_embedding_type=TIME_EMBEDDING_TYPE,
        state_embedding_type="linear",
        #n_ctx=args.n_ctx,
        n_ctx=context_len,
        device=run_config.device
    )
    




   
   
    run_decision_transformer(
        run_config=run_config,
        transformer_config=transformer_model_config,
        offline_config=offline_config,
        history_dataset=history_dataset,
    )
