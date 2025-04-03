# Explanation of "Flexible Motion In-betweening with Diffusion Models" (CondMDI)

## Overview

## Base Architecture: Motion Diffusion Model (MDM)

Before explaining CondMDI, it's important to understand its foundation - the Motion Diffusion Model (MDM):

1. **Core Architecture**
   - Transformer-based encoder-only architecture
   - Uses a diffusion process for motion generation
   - Key components:
     * Input processing layer
     * Positional encoding
     * Transformer encoder
     * Output processing layer
     * Text encoder (CLIP or BERT)

2. **Key Features**
   - Predicts the clean sample directly (rather than noise)
   - Supports multiple conditioning modes (text, action, or none)
   - Uses geometric losses for better motion quality:
     * Position loss
     * Foot contact loss
     * Velocity loss

3. **Training Process**
   - Uses a denoising diffusion process
   - Implements classifier-free guidance
   - Supports multiple motion representations (rotations or positions)

## Technical Implementation Details of MDM and CondMDI

## Motion Diffusion Model (MDM) Implementation

### Data Loading and Preprocessing

1. **Motion Data Format**
   - Input: Motion sequences represented as joint rotations or positions
   - Each frame contains:
     * Root joint position (3D coordinates)
     * Root joint rotation (quaternion or 6D rotation)
     * Joint rotations for each body joint (quaternion or 6D rotation)
   - Common formats:
     * AMASS dataset format
     * HumanML3D format
     * KIT format

2. **Data Preprocessing**
   - Normalization:
     * Center root joint positions
     * Normalize joint rotations
     * Scale motion sequences to consistent length
   - Feature extraction:
     * Convert quaternions to 6D rotation representation if needed
     * Compute joint velocities
     * Extract foot contact information

### Model Architecture

1. **Core Components**
   ```python
   class MDM(nn.Module):
       def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                    latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                    activation="gelu", data_rep='rot6d', dataset='amass', clip_dim=512):
           # Input processing
           self.input_process = InputProcess(data_rep, input_feats, latent_dim)
           
           # Positional encoding
           self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
           
           # Transformer encoder
           seqTransEncoderLayer = nn.TransformerEncoderLayer(
               d_model=latent_dim,
               nhead=num_heads,
               dim_feedforward=ff_size,
               dropout=dropout,
               activation=activation
           )
           self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers)
           
           # Text conditioning (CLIP or BERT)
           if 'text' in cond_mode:
               self.clip_model = self.load_and_freeze_clip(clip_version)
               self.embed_text = nn.Linear(clip_dim, latent_dim)
           
           # Output processing
           self.output_process = OutputProcess(data_rep, input_feats, latent_dim, njoints, nfeats)
   ```

2. **Input Processing**
   - Converts raw motion data to latent space representation
   - Handles different motion representations (rotations/positions)
   - Adds positional encoding for temporal information

3. **Transformer Architecture**
   - Encoder-only architecture
   - Multi-head attention mechanism
   - Position-wise feed-forward networks
   - Layer normalization and residual connections

### Training Process

1. **Diffusion Process Setup**
   ```python
   def setup_diffusion(self):
       # Define noise schedule
       self.betas = self.get_noise_schedule()
       self.alphas = 1 - self.betas
       self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
       
       # Define diffusion steps
       self.num_timesteps = len(self.betas)
   ```

2. **Training Loop**
   ```python
   def train_step(self, batch):
       # 1. Prepare input data
       x = batch['motion']  # [batch_size, njoints, nfeats, nframes]
       text = batch['text']  # Optional text conditioning
       
       # 2. Sample timestep
       t = torch.randint(0, self.num_timesteps, (x.shape[0],))
       
       # 3. Add noise to clean motion
       noise = torch.randn_like(x)
       x_t = self.q_sample(x, t, noise)
       
       # 4. Forward pass
       model_output = self.forward(x_t, t, text)
       
       # 5. Compute losses
       loss = self.compute_loss(model_output, x, noise)
       
       return loss
   ```

3. **Loss Functions**
   ```python
   def compute_loss(self, model_output, target, noise):
       # Simple loss (MSE between predicted and target)
       simple_loss = F.mse_loss(model_output, target)
       
       # Geometric losses
       pos_loss = self.compute_position_loss(model_output, target)
       foot_loss = self.compute_foot_contact_loss(model_output)
       vel_loss = self.compute_velocity_loss(model_output, target)
       
       # Total loss
       total_loss = (simple_loss + 
                    self.lambda_pos * pos_loss + 
                    self.lambda_foot * foot_loss + 
                    self.lambda_vel * vel_loss)
       
       return total_loss
   ```

4. **Geometric Losses**
   - Position loss: Ensures correct joint positions
   - Foot contact loss: Prevents foot sliding
   - Velocity loss: Maintains smooth motion

### Inference Process

1. **Sampling**
   ```python
   def sample(self, text=None, action=None, num_frames=60):
       # 1. Initialize with noise
       x = torch.randn(1, self.njoints, self.nfeats, num_frames)
       
       # 2. Iterative denoising
       for t in reversed(range(self.num_timesteps)):
           # Predict clean sample
           model_output = self.forward(x, t, text, action)
           
           # Update sample
           x = self.p_sample(model_output, x, t)
       
       return x
   ```

2. **Conditioning**
   - Text conditioning through CLIP embeddings
   - Action conditioning through action embeddings
   - Classifier-free guidance for better control

## CondMDI Implementation

### Keyframe Conditioning

1. **Masking Mechanism**
   ```python
   def create_keyframe_mask(self, motion, keyframes):
       # Create binary mask indicating keyframe positions
       mask = torch.zeros_like(motion)
       for frame_idx, joint_mask in keyframes:
           mask[:, :, :, frame_idx] = joint_mask
       return mask
   ```

2. **Training with Keyframes**
   ```python
   def train_step_with_keyframes(self, batch):
       # 1. Prepare data
       x = batch['motion']
       keyframes = batch['keyframes']
       text = batch['text']
       
       # 2. Create keyframe mask
       mask = self.create_keyframe_mask(x, keyframes)
       
       # 3. Sample timestep and add noise
       t = torch.randint(0, self.num_timesteps, (x.shape[0],))
       noise = torch.randn_like(x)
       x_t = self.q_sample(x, t, noise)
       
       # 4. Forward pass with mask
       model_output = self.forward(x_t, t, text, mask=mask)
       
       # 5. Compute losses
       loss = self.compute_loss(model_output, x, noise, mask)
       
       return loss
   ```

3. **Keyframe Loss**
   ```python
   def compute_keyframe_loss(self, model_output, target, mask):
       # Only compute loss on non-keyframe positions
       masked_output = model_output * (1 - mask)
       masked_target = target * (1 - mask)
       return F.mse_loss(masked_output, masked_target)
   ```

### Architecture Modifications

1. **Keyframe Encoder**
   ```python
   class KeyframeEncoder(nn.Module):
       def __init__(self, latent_dim):
           super().__init__()
           self.encoder = nn.Sequential(
               nn.Linear(latent_dim, latent_dim),
               nn.ReLU(),
               nn.Linear(latent_dim, latent_dim)
           )
           
       def forward(self, keyframes, mask):
           # Encode keyframe information
           encoded = self.encoder(keyframes)
           return encoded * mask
   ```

2. **Modified Transformer**
   - Additional attention heads for keyframe information
   - Masked attention mechanism
   - Keyframe-aware positional encoding

### Inference with Keyframes

1. **Keyframe-Guided Sampling**
   ```python
   def sample_with_keyframes(self, keyframes, text=None, num_frames=60):
       # 1. Initialize with noise
       x = torch.randn(1, self.njoints, self.nfeats, num_frames)
       
       # 2. Set keyframe values
       mask = self.create_keyframe_mask(x, keyframes)
       x = x * (1 - mask) + keyframes * mask
       
       # 3. Iterative denoising
       for t in reversed(range(self.num_timesteps)):
           # Predict clean sample
           model_output = self.forward(x, t, text, mask=mask)
           
           # Update sample while preserving keyframes
           x = self.p_sample(model_output, x, t, mask)
       
       return x
   ```

## Methodology

1. **Training Strategy**
   - Uses a masked conditional diffusion model architecture
   - During training, randomly samples:
     * Keyframe positions in time
     * Which joints to constrain at each keyframe
     * The number of keyframes to use
   - Implements a masking mechanism that:
     * Indicates which frames and joints are observed (keyframes)
     * Specifies which frames and joints need to be generated
     * Allows for partial keyframe constraints (some joints specified, others free)
   - Training process:
     * Takes a complete motion sequence as input
     * Randomly masks out frames and joints to create keyframe constraints
     * Learns to reconstruct the full motion given these constraints
     * Uses a denoising diffusion process to generate missing frames

2. **Architecture**
   - Based on a transformer-based diffusion model
   - Key components:
     * Motion encoder: Processes the input motion sequence
     * Keyframe encoder: Handles the keyframe constraints
     * Mask encoder: Processes the masking information
     * Denoising network: Generates the full motion sequence
   - Handles both spatial and temporal constraints:
     * Spatial: Joint positions and orientations
     * Temporal: Frame timing and sequence length
   - Global root handling:
     * Uses absolute root joint positions and orientations
     * Avoids issues with relative-to-previous-frame representation
     * Enables better handling of sparse keyframes

3. **Keyframe Handling**
   - Flexible keyframe specification:
     * Supports arbitrary number of keyframes
     * Allows keyframes at any position in the sequence
     * Can handle partial keyframes (some joints specified)
   - Constraint satisfaction:
     * Uses a masking mechanism to enforce keyframe constraints
     * Maintains exact keyframe values during generation
     * Allows for smooth interpolation between keyframes
   - Quality preservation:
     * Maintains motion naturalness even with sparse keyframes
     * Avoids common artifacts like foot sliding
     * Preserves motion style and characteristics

4. **Inference Process**
   - Takes as input:
     * Keyframe constraints (positions and timing)
     * Optional text prompt for style guidance
     * Mask indicating which frames/joints to generate
   - Generation steps:
     * Initializes with random noise
     * Iteratively denoises the sequence
     * Enforces keyframe constraints at each step
     * Maintains coherence with text prompts
   - Output:
     * Complete motion sequence
     * Exact match with keyframe constraints
     * Natural-looking interpolation between keyframes

5. **Technical Innovations**
   - Unified approach:
     * Single model handles all types of keyframe constraints
     * No need for separate modules or post-processing
   - Efficient training:
     * Random sampling of constraints during training
     * Enables handling of various constraint patterns
   - Fast inference:
     * Direct generation without iterative refinement
     * No need for additional guidance or optimization