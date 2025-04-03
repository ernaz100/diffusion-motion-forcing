import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

class HumanML3DDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        max_motion_length: int = 196,
        min_motion_length: int = 24,
        window_size: int = 60,
        fps: int = 20,
        unit_length: int = 4,
        use_text: bool = True,
        use_abs_3d: bool = True,  # Whether to use absolute 3D coordinates
        verbose: bool = False  # Add verbose flag
    ):
        """
        Initialize the HumanML3D dataset loader.
        
        Args:
            data_root: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            max_motion_length: Maximum length of motion sequences
            min_motion_length: Minimum length of motion sequences
            window_size: Size of the motion window
            fps: Frames per second
            unit_length: Unit length for motion representation
            use_text: Whether to use text descriptions
            use_abs_3d: Whether to use absolute 3D coordinates
            verbose: Whether to print transformation steps
        """
        self.data_root = data_root
        self.split = split
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.window_size = window_size
        self.fps = fps
        self.unit_length = unit_length
        self.use_text = use_text
        self.use_abs_3d = use_abs_3d
        self.verbose = verbose
        self._first_sample = True  # Add flag for first sample
        
        # Load mean and std for normalization
        if use_abs_3d:
            self.mean = np.load(os.path.join(data_root, 'Mean_abs_3d.npy'))
            self.std = np.load(os.path.join(data_root, 'Std_abs_3d.npy'))
        else:
            self.mean = np.load(os.path.join(data_root, 'Mean.npy'))
            self.std = np.load(os.path.join(data_root, 'Std.npy'))
        
        # Load motion data paths
        self._load_motion_paths()
        
        # Load text data if needed
        if self.use_text:
            self._load_text_data()
            
        # Initialize batch counter
        self._current_batch = 0
        self._samples_in_batch = 0
    
    def _load_motion_paths(self):
        """Load motion file paths and filter by length."""
        # Read split file
        split_file = os.path.join(self.data_root, f'{self.split}.txt')
        with open(split_file, 'r') as f:
            motion_files = [line.strip() for line in f.readlines()]
            
        self.motion_files = motion_files
        
        # Store paths for both joints and joint vectors
        joints_dir = 'new_joints_abs_3d' if self.use_abs_3d else 'new_joints'
        vecs_dir = 'new_joint_vecs_abs_3d' if self.use_abs_3d else 'new_joint_vecs'
        
        self.joint_paths = [os.path.join(self.data_root, joints_dir, f'{f}.npy') for f in motion_files]
        self.vec_paths = [os.path.join(self.data_root, vecs_dir, f'{f}.npy') for f in motion_files]
    
    def _load_text_data(self):
        """Load text descriptions for motions."""
        # Read text descriptions
        texts_dir = os.path.join(self.data_root, 'texts')
        self.text_data = []
        for motion_file in self.motion_files:
            text_path = os.path.join(texts_dir, f'{motion_file}.txt')
            with open(text_path, 'r') as f:
                texts = f.readlines()
            self.text_data.append(texts)
    
    def _process_motion(self, motion_file):
        # Only print for the first sample ever
        if self.verbose and self._first_sample:
            print(f"\nProcessing first sample: {motion_file}")
        
        # Load joint positions and vectors
        if self.use_abs_3d:
            joints = np.load(os.path.join(self.data_root, 'new_joints_abs_3d', f'{motion_file}.npy'))
            vecs = np.load(os.path.join(self.data_root, 'new_joint_vecs_abs_3d', f'{motion_file}.npy'))
        else:
            joints = np.load(os.path.join(self.data_root, 'new_joints', f'{motion_file}.npy'))
            vecs = np.load(os.path.join(self.data_root, 'new_joint_vecs', f'{motion_file}.npy'))
        
        if self.verbose and self._first_sample:
            print(f"Initial shapes - joints: {joints.shape}, vectors: {vecs.shape}")
            print(f"Initial stats - joints: min={joints.min():.4f}, max={joints.max():.4f}, mean={joints.mean():.4f}, std={joints.std():.4f}")
            print(f"Initial stats - vectors: min={vecs.min():.4f}, max={vecs.max():.4f}, mean={vecs.mean():.4f}, std={vecs.std():.4f}")

        # Normalize joints - shape is [seq_len, njoints, 3]
        # Flatten to [seq_len, njoints * 3] for normalization
        seq_len, njoints, _ = joints.shape
        joints_flat = joints.reshape(seq_len, -1)  # [seq_len, njoints * 3]
        if self.verbose and self._first_sample:
            print(f"After flattening joints: {joints_flat.shape}")
        
        joints_flat = (joints_flat - self.mean[:njoints * 3]) / self.std[:njoints * 3]
        if self.verbose and self._first_sample:
            print(f"After normalizing joints - min: {joints_flat.min():.4f}, max: {joints_flat.max():.4f}, mean: {joints_flat.mean():.4f}, std: {joints_flat.std():.4f}")
        
        joints = joints_flat.reshape(seq_len, njoints, 3)  # Back to [seq_len, njoints, 3]
        if self.verbose and self._first_sample:
            print(f"After reshaping joints back: {joints.shape}")

        # Normalize vectors - shape is [seq_len, 263]
        vecs = (vecs - self.mean) / self.std
        if self.verbose and self._first_sample:
            print(f"After normalizing vectors - min: {vecs.min():.4f}, max: {vecs.max():.4f}, mean: {vecs.mean():.4f}, std: {vecs.std():.4f}")

        # Convert to torch tensors and reshape
        joints = torch.from_numpy(joints).float()  # [seq_len, njoints, 3]
        vecs = torch.from_numpy(vecs).float()  # [seq_len, 263]
        if self.verbose and self._first_sample:
            print(f"After converting to torch - joints: {joints.shape}, vectors: {vecs.shape}")

        # Pad or truncate to window_size
        if joints.shape[0] < self.window_size:
            # Pad
            pad_len = self.window_size - joints.shape[0]
            joints = torch.cat([joints, joints[-1:].repeat(pad_len, 1, 1)], dim=0)
            vecs = torch.cat([vecs, vecs[-1:].repeat(pad_len, 1)], dim=0)
            if self.verbose and self._first_sample:
                print(f"After padding to {self.window_size} - joints: {joints.shape}, vectors: {vecs.shape}")
        else:
            # Truncate
            joints = joints[:self.window_size]
            vecs = vecs[:self.window_size]
            if self.verbose and self._first_sample:
                print(f"After truncating to {self.window_size} - joints: {joints.shape}, vectors: {vecs.shape}")

        # Reshape for model input
        joints = joints.permute(1, 2, 0)  # [njoints, 3, seq_len]
        vecs = vecs.unsqueeze(0).permute(0, 2, 1)  # [1, 263, seq_len]
        if self.verbose and self._first_sample:
            print(f"Final shapes - joints: {joints.shape}, vectors: {vecs.shape}")
            self._first_sample = False  # Disable printing for subsequent samples

        return joints, vecs
    
    def _process_text(self, texts: List[str]) -> str:
        """Process text descriptions."""
        # For now, just return a random text description
        return np.random.choice(texts).strip()
    
    def __len__(self) -> int:
        return len(self.motion_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single data sample.
        
        Returns:
            Dictionary containing:
            - joints: Joint positions tensor [njoints, 3, seq_len]
            - vectors: Joint vectors tensor [1, 263, seq_len]
            - text: Text description (if use_text=True)
            - length: Original motion length
        """
        motion_file = self.motion_files[idx]
        joints, vectors = self._process_motion(motion_file)
        
        # Get sequence length from the original motion file
        joint_path = os.path.join(
            self.data_root, 
            'new_joints_abs_3d' if self.use_abs_3d else 'new_joints',
            f'{motion_file}.npy'
        )
        orig_length = np.load(joint_path).shape[0]
        
        sample = {
            'joints': joints,
            'vectors': vectors,
            'length': orig_length
        }
        
        if self.use_text:
            sample['text'] = self._process_text(self.text_data[idx])
            
        return sample

def get_dataloader(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    split: str = 'train',
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the HumanML3D dataset.
    
    Args:
        data_root: Root directory containing the dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        split: Dataset split ('train', 'val', 'test')
        **kwargs: Additional arguments for HumanML3DDataset
        
    Returns:
        DataLoader instance
    """
    dataset = HumanML3DDataset(data_root=data_root, split=split, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    ) 