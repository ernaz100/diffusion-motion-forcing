"""
Configuration file for MDM model and training.
Original implementation reference: motion-diffusion-model/configs/config.py
"""

# Data configuration
import torch


DATA_CONFIG = {
    "data_root": "data/humanml3d",  # Path to HumanML3D dataset
    "batch_size": 32,
    "num_workers": 4,
    "split": "train",  # train, val, test
    "window_size": 60  # Number of frames in each motion sequence
}

# Model configuration
MODEL_CONFIG = {
    "njoints": 22,  # Number of joints in HumanML3D
    "nfeats": 263,  # Dimension of joint vectors (4 root + 63 ric + 126 rot + 66 vel + 4 foot)
    "latent_dim": 256,
    "ff_size": 1024,
    "num_layers": 8,
    "num_heads": 4,
    "dropout": 0.1,
    "activation": "gelu",
    "clip_version": "ViT-B/32"
}

# Training configuration
TRAIN_CONFIG = {
    "save_dir": "checkpoints/mdm",  # Directory to save checkpoints
    "num_epochs": 1000,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_interval": 100,  # Log every N batches
    "save_interval": 1000,  # Save checkpoint every N batches
    "resume_checkpoint": None,  # Path to checkpoint to resume from
    
    # Noise scheduler configuration
    "num_timesteps": 1000,  # Number of diffusion timesteps
    "beta_start": 0.0001,  # Starting beta value
    "beta_end": 0.02,  # Ending beta value
    "rescale_timesteps": False  # Whether to rescale timesteps to [0, 1000]
}

# Missing features compared to original implementation:
# 1. No support for different architectures (trans_enc, trans_dec, gru)
# 2. No support for different pose representations (rot6d, etc.)
# 3. No support for different datasets (amass, etc.)
# 4. No support for different text encoders (BERT)
# 5. No support for action conditioning
# 6. No support for target conditioning
# 7. No support for prefix completion
# 8. No support for frame masking
# 9. No support for different embedding policies
# 10. No support for different normalization strategies
# 11. No support for distributed training
# 12. No support for mixed precision training
# 13. No support for EMA (Exponential Moving Average)
# 14. No support for different training platforms (WandB, Tensorboard, etc.)
# 15. No support for evaluation during training
# 16. No support for different loss functions
# 17. No support for different learning rate schedules
# 18. No support for gradient accumulation
# 19. No support for different sampling strategies
# 20. No support for different validation metrics