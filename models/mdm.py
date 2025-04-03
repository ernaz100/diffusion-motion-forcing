"""
MDM model implementation.
Original implementation reference: motion-diffusion-model/model/mdm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Dict, Optional
import math

# Original implementation reference: motion-diffusion-model/model/mdm.py

# Features adapted from original implementation:
# 1. CLIP text encoder with linear projection to latent dimension
# 2. Transformer encoder architecture with positional encoding
# 3. Timestep embedding 
# 4. Input and output processing layers
# 5. Basic conditioning through text embeddings
# 6. Batch-first transformer implementation
# 7. Dropout and activation functions
# 8. Model parameter configuration matching original


# Missing features compared to original:
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
class MDM(nn.Module):
    def __init__(
        self,
        njoints: int = 22,  # Number of joints in HumanML3D
        nfeats: int = 263,  # Dimension of joint vectors (4 root + 63 ric + 126 rot + 66 vel + 4 foot)
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        clip_version: str = "ViT-B/32",
    ):
        super().__init__()
        
        # Original reference: motion-diffusion-model/model/mdm.py:MDM.__init__
        self.njoints = njoints
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        
        # Input processing - Original reference: motion-diffusion-model/model/mdm.py:InputProcess
        self.input_process = InputProcess(nfeats, latent_dim)
        
        # Positional encoding - Original reference: motion-diffusion-model/model/mdm.py:PositionalEncoding
        self.position_encoding = PositionalEncoding(latent_dim, dropout)
        
        # Transformer encoder - Original reference: motion-diffusion-model/model/mdm.py:MDM.__init__ (trans_enc branch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Timestep embedding - Original reference: motion-diffusion-model/model/mdm.py:TimestepEmbedder
        self.timestep_embedder = TimestepEmbedder(latent_dim, self.position_encoding)
        
        # Text encoder - Original reference: motion-diffusion-model/model/mdm.py:MDM.load_and_freeze_clip
        self.clip_version = clip_version
        self.clip_model = self._load_clip()
        # Project CLIP embeddings to latent dimension - Original reference: motion-diffusion-model/model/mdm.py:MDM.__init__
        self.embed_text = nn.Linear(512, latent_dim)  # CLIP ViT-B/32 outputs 512-dimensional embeddings
        
        # Output processing - Original reference: motion-diffusion-model/model/mdm.py:OutputProcess
        self.output_process = OutputProcess(nfeats, latent_dim)

    def _load_clip(self):
        # Original reference: motion-diffusion-model/model/mdm.py:MDM.load_and_freeze_clip
        import clip
        model, _ = clip.load(self.clip_version, device="cpu", jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def clip_encode_text(self, text: str) -> torch.Tensor:
        # Original reference: motion-diffusion-model/model/mdm.py:MDM.clip_encode_text
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            # Project CLIP embeddings to latent dimension - Original reference: motion-diffusion-model/model/mdm.py:MDM.forward
            text_features = self.embed_text(text_features)
        return text_features

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, 1, nfeats, seq_len]
        timesteps: torch.Tensor,  # [batch_size]
        text: Optional[str] = None,  # Text description
    ) -> torch.Tensor:
        """
        Original reference: motion-diffusion-model/model/mdm.py:MDM.forward
        """
        # Process input
        x = self.input_process(x)  # [batch_size, seq_len, latent_dim]
        
        # Add positional encoding
        x = self.position_encoding(x)
        
        # Add timestep embedding
        t_emb = self.timestep_embedder(timesteps)  # [batch_size, latent_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # [batch_size, seq_len, latent_dim]
        x = x + t_emb
        
        # Add text embedding if provided
        if text is not None:
            text_emb = self.clip_encode_text(text)  # [batch_size, latent_dim]
            text_emb = text_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # [batch_size, seq_len, latent_dim]
            x = x + text_emb
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, seq_len, latent_dim]
        
        # Process output
        x = self.output_process(x)  # [batch_size, seq_len, nfeats]
        
        return x

class InputProcess(nn.Module):
    # Original reference: motion-diffusion-model/model/mdm.py:InputProcess
    def __init__(self, nfeats: int, latent_dim: int):
        super().__init__()
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.pose_embedding = nn.Linear(nfeats, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1, nfeats, seq_len]
        x = x.squeeze(1)  # [batch_size, nfeats, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, nfeats]
        x = self.pose_embedding(x)  # [batch_size, seq_len, latent_dim]
        return x

class PositionalEncoding(nn.Module):
    # Original reference: motion-diffusion-model/model/mdm.py:PositionalEncoding
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    """
    Timestep embedding module.
    Original reference: motion-diffusion-model/model/mdm.py:TimestepEmbedder
    
    This module takes timestep indices and converts them into embeddings that can be used by the model.
    The process is:
    1. Use the pre-computed positional encoding table from sequence_pos_encoder
    2. Index into the table using the timestep indices
    3. Pass through linear layers to get the final embedding
    
    The original implementation reuses the same positional encoding table for both sequence positions
    and timesteps, which is why we pass sequence_pos_encoder to this class.
    """
    def __init__(self, latent_dim: int, sequence_pos_encoder: PositionalEncoding):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        # Create a simple MLP to process the timestep embeddings
        # The original implementation uses two linear layers with SiLU activation
        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),  # First linear layer
            nn.SiLU(),  # SiLU activation (Swish)
            nn.Linear(time_embed_dim, time_embed_dim),  # Second linear layer
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Convert timestep indices into embeddings.
        
        Args:
            timesteps: Tensor of shape [batch_size] containing timestep indices
            
        Returns:
            Tensor of shape [batch_size, latent_dim] containing the timestep embeddings
        """
        # Convert timesteps to long for indexing
        timesteps = timesteps.long()
        
        # Get embeddings from the pre-computed table
        # sequence_pos_encoder.pe has shape [max_len, 1, latent_dim]
        # timesteps has shape [batch_size]
        # We need to index into the first dimension of pe
        pe = self.sequence_pos_encoder.pe[timesteps]  # [batch_size, 1, latent_dim]
        
        # Process through MLP and remove the middle dimension
        return self.time_embed(pe.squeeze(1))  # [batch_size, latent_dim]

class OutputProcess(nn.Module):
    # Original reference: motion-diffusion-model/model/mdm.py:OutputProcess
    def __init__(self, nfeats: int, latent_dim: int):
        super().__init__()
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.pose_embedding = nn.Linear(latent_dim, nfeats)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, latent_dim]
        x = self.pose_embedding(x)  # [batch_size, seq_len, nfeats]
        return x 