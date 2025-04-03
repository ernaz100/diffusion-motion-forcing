"""
Training script for MDM model.
Original implementation reference: motion-diffusion-model/train/train_mdm.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from typing import Dict, Any

from models.mdm import MDM
from utils.noise_scheduler import NoiseScheduler, ModelMeanType, ModelVarType, LossType
from data_loaders.humanml3d_loader import get_dataloader
from configs.mdm_config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

def train(single_iteration: bool = False) -> None:
    """
    Train the MDM model using the configuration from mdm_config.py.
    # Original implementation reference: motion-diffusion-model/train/train_mdm.py
    # Missing features compared to original:
    # 1. No support for distributed training
    # 2. No support for mixed precision training
    # 3. No support for EMA (Exponential Moving Average)
    # 4. No support for different training platforms (WandB, Tensorboard, etc.)
    # 5. No support for evaluation during training
    # 6. No support for different loss functions
    # 7. No support for different learning rate schedules
    # 8. No support for gradient accumulation
    # 9. No support for different sampling strategies
    # 10. No support for different validation metrics
    """
    # Create save directory
    os.makedirs(TRAIN_CONFIG["save_dir"], exist_ok=True)
    
    # Save config
    config = {
        "data_config": DATA_CONFIG,
        "model_config": MODEL_CONFIG,
        "train_config": TRAIN_CONFIG
    }
    with open(os.path.join(TRAIN_CONFIG["save_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Initialize data loader
    dataloader = get_dataloader(
        data_root=DATA_CONFIG["data_root"],
        batch_size=DATA_CONFIG["batch_size"],
        num_workers=DATA_CONFIG["num_workers"],
        split=DATA_CONFIG["split"],
        window_size=DATA_CONFIG["window_size"],
        use_text=True,
        use_abs_3d=True,
        verbose=False
    )
    
    # Initialize model
    model = MDM(
        njoints=MODEL_CONFIG["njoints"],
        nfeats=MODEL_CONFIG["nfeats"],
        latent_dim=MODEL_CONFIG["latent_dim"],
        ff_size=MODEL_CONFIG["ff_size"],
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        dropout=MODEL_CONFIG["dropout"],
        activation=MODEL_CONFIG["activation"],
        clip_version=MODEL_CONFIG["clip_version"]
    ).to(TRAIN_CONFIG["device"])
    
    # Initialize noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=TRAIN_CONFIG["num_timesteps"],
        beta_start=TRAIN_CONFIG["beta_start"],
        beta_end=TRAIN_CONFIG["beta_end"],
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=TRAIN_CONFIG["rescale_timesteps"]
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    
    # Load checkpoint if provided
    if TRAIN_CONFIG["resume_checkpoint"] is not None:
        checkpoint = torch.load(TRAIN_CONFIG["resume_checkpoint"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
    
    # Training loop
    for epoch in range(start_epoch, TRAIN_CONFIG["num_epochs"]):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # Move batch to device
            vectors = batch["vectors"].to(TRAIN_CONFIG["device"])
            text = batch["text"]
            
            # Sample random timesteps
            t = torch.randint(
                0, noise_scheduler.num_timesteps, (vectors.shape[0],), device=vectors.device
            ).long()
            
            # Add noise to the input
            noise = torch.randn_like(vectors)
            noisy_vectors = noise_scheduler.q_sample(vectors, t, noise)
            
            # Forward pass
            optimizer.zero_grad()
            noise_pred = model(noisy_vectors, t, text)
            
            # Reshape noise to match model output
            # noise: [batch_size, 1, nfeats, seq_len] -> [batch_size, seq_len, nfeats]
            noise = noise.squeeze(1).permute(0, 2, 1)
            
            # Compute loss
            loss = nn.MSELoss()(noise_pred, noise)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % TRAIN_CONFIG["log_interval"] == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
            
            # Save checkpoint
            if batch_idx % TRAIN_CONFIG["save_interval"] == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item()
                }
                torch.save(
                    checkpoint,
                    os.path.join(TRAIN_CONFIG["save_dir"], f"checkpoint_epoch{epoch}_batch{batch_idx}.pt")
                )
            
            # If this is a dry run, break after one iteration
            if single_iteration:
                print("\nDry run completed. Breaking after one iteration.")
                return
        
        # Print epoch summary
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train() 