"""
Inference script for MDM model.
Original implementation reference: motion-diffusion-model/sample/generate.py
"""

import torch
import numpy as np
from models.mdm import MDM
from utils.noise_scheduler import NoiseScheduler, ModelMeanType, ModelVarType, LossType
from data_loaders.humanml3d_loader import get_dataloader
from configs.mdm_config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

class MDMInference:
    """
    Class for generating motion samples from text descriptions using the MDM model.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Initialize the MDM inference model.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run the model on (cuda or cpu)
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value for noise schedule
            beta_end: Ending beta value for noise schedule
        """
        self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.model = MDM(
            njoints=MODEL_CONFIG["njoints"],
            nfeats=MODEL_CONFIG["nfeats"],
            latent_dim=MODEL_CONFIG["latent_dim"],
            ff_size=MODEL_CONFIG["ff_size"],
            num_layers=MODEL_CONFIG["num_layers"],
            num_heads=MODEL_CONFIG["num_heads"],
            dropout=MODEL_CONFIG["dropout"],
            activation=MODEL_CONFIG["activation"],
            clip_version=MODEL_CONFIG["clip_version"]
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=True
        )

    def sample(
        self,
        text: str,
        batch_size: int = 1,
        seq_len: int = 60,
        nfeats: int = 263,
        return_tensor: bool = False,
    ) -> np.ndarray:
        """
        Generate motion samples from a text description.
        
        Args:
            text: Text description of the motion
            batch_size: Number of samples to generate
            seq_len: Length of the motion sequence
            nfeats: Number of features per frame
            return_tensor: If True, return PyTorch tensor instead of numpy array
            
        Returns:
            Generated motion samples as numpy array or PyTorch tensor
        """
        shape = (batch_size, 1, nfeats, seq_len)
        
        # Generate samples
        samples = self.noise_scheduler.p_sample_loop(
            model=self.model,
            shape=shape,
            text=text,
            progress=True
        )
        
        if return_tensor:
            return samples
        return samples.cpu().numpy()

    def print_motion_stats(
        self,
        motion_data: np.ndarray,
    ):
        """
        Print statistics about the generated motion data.
        
        Args:
            motion_data: Motion data of shape [batch_size, 1, nfeats, seq_len]
        """
        # Get the first sample and remove batch dimension
        motion = motion_data[0, 0]  # [nfeats, seq_len]
        
        print("\n=== Motion Statistics ===")
        print(f"Shape: {motion.shape}")
        print(f"Number of features: {motion.shape[0]}")
        print(f"Sequence length: {motion.shape[1]}")
        
def main():
    # Example usage
    model_path = "./checkpoints/mdm/checkpoint_epoch0_batch0.pt" 
    inference = MDMInference(model_path)
    
    # Generate samples
    text = "a person walking forward"
    samples = inference.sample(text, batch_size=1)
    
    # Print statistics
    inference.print_motion_stats(samples)

if __name__ == "__main__":
    main() 