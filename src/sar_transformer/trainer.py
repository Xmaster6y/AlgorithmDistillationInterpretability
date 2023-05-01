import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

import wandb
from src.config import EnvironmentConfig
from src.models.trajectory_transformer import (
    AlgorithmDistillationTransformer,
    TrajectoryTransformer,
)

from .dataset import create_history_dataloader
from .eval import evaluate_ad_agent


def train(
    model,
    train_dataloader,
    env_config,
    lr=3e-4,
    clip=1.,
    device="cuda",
    track=False,
    train_epochs=100,
    eval_frequency=5,
    eval_length=100,
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

            pbar.set_description(f"TRAIN - Epoch {epoch+1}: {loss.item():.4f}")
            pbar.update(1)
            
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
        if (epoch+1) % eval_frequency == 0:
            # Evaluate the performance of the model on the new env
            evaluate_ad_agent(
                model=model,
                env_config=env_config,
                n_episodes=eval_length,
                temp=1,
                device=device,
                track=track
             )

    return model
