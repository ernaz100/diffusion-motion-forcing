import sys
import os
import torch
import numpy as np
import json
from pathlib import Path
import torch.nn as nn
from models.mdm import MDM
from utils.noise_scheduler import NoiseScheduler
from torch.utils.data import DataLoader

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from data_loaders.humanml3d_loader import get_dataloader
from configs.mdm_config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from training.train_mdm import train

def inspect_batch(batch):
    """Print detailed information about a batch of data."""
    print("\n=== Batch Information ===")
    print(f"Number of samples in batch: {len(batch['joints'])}")
    
    # Joint data
    joints = batch['joints']
    print("\nJoint data:")
    print(f"Shape: {joints.shape}")  # Should be [batch_size, njoints, 3, seq_len]
    print(f"Type: {joints.dtype}")
    print(f"Min value: {joints.min().item():.4f}")
    print(f"Max value: {joints.max().item():.4f}")
    print(f"Mean value: {joints.mean().item():.4f}")
    print(f"Std value: {joints.std().item():.4f}")
    
    # Print joint information
    print("\nJoint information:")
    print(f"Number of joints: {joints.shape[1]}")
    print(f"Features per joint: {joints.shape[2]}")  # Should be 3 (x, y, z)
    print(f"Sequence length: {joints.shape[3]}")
    
    # Vector data
    vectors = batch['vectors']
    print("\nVector data:")
    print(f"Shape: {vectors.shape}")  # Should be [batch_size, 1, 263, seq_len]
    print(f"Type: {vectors.dtype}")
    print(f"Min value: {vectors.min().item():.4f}")
    print(f"Max value: {vectors.max().item():.4f}")
    print(f"Mean value: {vectors.mean().item():.4f}")
    print(f"Std value: {vectors.std().item():.4f}")
    
    # Length information
    lengths = batch['length']
    print("\nMotion lengths:")
    print(f"Shape: {lengths.shape}")
    print(f"Min length: {lengths.min().item()}")
    print(f"Max length: {lengths.max().item()}")
    print(f"Mean length: {lengths.float().mean().item():.2f}")
    
    # Text descriptions (if available)
    if 'text' in batch:
        print("\nText descriptions:")
        for i, text in enumerate(batch['text'][:3]):  # Show first 3 texts
            print(f"Sample {i}: {text}")
    
    # Print memory usage
    print("\nMemory usage:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.element_size() * value.nelement() / 1024 / 1024:.2f} MB")

def check_data_directory(data_root):
    """Check if the data directory structure is correct."""
    print("\n=== Checking Data Directory Structure ===")
    
    # Check main directories
    required_dirs = [
        'new_joints', 'new_joint_vecs',
        'new_joints_abs_3d', 'new_joint_vecs_abs_3d',
        'texts'
    ]
    for dir_name in required_dirs:
        dir_path = os.path.join(data_root, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name} directory exists")
            # Count files in directory
            files = [f for f in os.listdir(dir_path) if f.endswith(('.npy', '.txt'))]
            print(f"  Number of files: {len(files)}")
        else:
            print(f"✗ {dir_name} directory missing")
    
    # Check split files
    split_files = ['train.txt', 'val.txt', 'test.txt']
    for split_file in split_files:
        split_path = os.path.join(data_root, split_file)
        if os.path.exists(split_path):
            print(f"✓ {split_file} exists")
            # Count lines
            with open(split_path, 'r') as f:
                lines = f.readlines()
                print(f"  Number of samples: {len(lines)}")
        else:
            print(f"✗ {split_file} missing")
            
    # Check mean/std files
    norm_files = ['Mean.npy', 'Std.npy', 'Mean_abs_3d.npy', 'Std_abs_3d.npy']
    for norm_file in norm_files:
        norm_path = os.path.join(data_root, norm_file)
        if os.path.exists(norm_path):
            print(f"✓ {norm_file} exists")
        else:
            print(f"✗ {norm_file} missing")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    try:
        # Initialize data loader
        dataloader = get_dataloader(
            data_root=DATA_CONFIG['data_root'],
            batch_size=DATA_CONFIG['batch_size'],
            num_workers=DATA_CONFIG['num_workers'],
            split=DATA_CONFIG['split'],
            window_size=DATA_CONFIG['window_size'],
            use_text=True,
            use_abs_3d=True,
            verbose=False
        )
        
        # Load one batch
        print("\nLoading one batch of data...")
        batch = next(iter(dataloader))
        
        # Inspect the batch
        inspect_batch(batch)
        
        # Run a single training iteration
        print("\nRunning a single training iteration...")
        train(single_iteration=True)
        
    except Exception as e:
        print(f"\nError during dry run: {str(e)}")
        print("\nConfiguration used:")
        print("DATA_CONFIG:", json.dumps(DATA_CONFIG, indent=2))
        print("MODEL_CONFIG:", json.dumps(MODEL_CONFIG, indent=2))
        print("TRAIN_CONFIG:", json.dumps(TRAIN_CONFIG, indent=2))
        raise

if __name__ == "__main__":
    main() 