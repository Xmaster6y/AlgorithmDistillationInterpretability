import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

import wandb
from config import EnvironmentConfig
from models.trajectory_transformer import (
    AlgorithmDistillationTransformer,
    TrajectoryTransformer,
)

from .dataset import create_history_dataloader
from .eval import evaluate_ad_agent


def train(
    model,
    train_dataloader,
    env,
    lr=3e-4,
    clip=1.,
    device="cuda",
    track=False,
    train_epochs=100,
    eval_frequency=10,
    eval_episodes=10,
    initial_rtg=[0.0, 1.0],
    eval_max_time_steps=100,
    eval_num_envs=8,
):
    # Create loss function and model optimizer
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs)
    train_batches_per_epoch = len(train_dataloader)
    
    # Start training
    for epoch in range(train_epochs):
        
        pbar = tqdm(total=train_batches_per_epoch)
        
        for batch, (s, a, r, ti) in enumerate(train_dataloader):
            total_batches = epoch * train_batches_per_epoch + batch

            model.train()

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(torch.float32)

            optimizer.zero_grad()
            
            states = s.to(device)
            action_labels = a.to(device)
            actions = action_labels[:, :-1]
            rewards = r.to(device)[:, :-1]
            time = ti.to(device)
            
            _, action_preds, _ = model.forward(
                states=states,
                actions=actions,
                rewards=rewards,
                timesteps=time,
            )
             
            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(action_labels, "b t i -> (b t i)").to(torch.int64)

            loss = loss_fn(action_preds, a_exp)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            pbar.update(1)
            pbar.set_description(f"Training AD, Epoch {epoch+1}: {loss.item():.4f}")

            if track:
                wandb.log({"train/loss": loss.item()}, step=total_batches)
                tokens_seen = (
                    (total_batches + 1)
                    * batch_size
                    * (model.transformer_config.n_ctx // 3)
                )
                wandb.log(
                    {"metrics/tokens_seen": tokens_seen}, step=total_batches
                )
        
        scheduler.step()
        pbar.close()
        
        # # at test frequency
        if epoch % eval_frequency == -1:
            # Evaluate the performance of the model on the new env
            eval_env_config = EnvironmentConfig(
                env_id=env.spec.id,
                capture_video=True,
                max_steps=min(
                    model.environment_config.max_steps, eval_max_time_steps
                ),
                fully_observed=False,
                one_hot_obs=(trajectory_data_set.observation_type == "one_hot"),
                view_size=env.observation_space["image"].shape[0]
                if "image" in list(env.observation_space.keys())
                else 7,
            )

            eval_env_func = make_env(
                config=eval_env_config,
                seed=batch,
                idx=0,
                run_name=f"dt_eval_videos_{batch}",
            )

            for rtg in initial_rtg:
                evaluate_dt_agent(
                    env_id=env.spec.id,
                    model=model,
                    env_func=eval_env_func,
                    trajectories=eval_episodes,
                    track=track,
                    batch_number=total_batches,
                    initial_rtg=float(rtg),
                    device=device,
                    num_envs=eval_num_envs,
                )

    return model
