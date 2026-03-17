"""
DDPM (Denoising Diffusion Probabilistic Models) Implementation
Based on Ho et al., 2020 (https://arxiv.org/abs/2006.11239)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings for timesteps.
    Used to encode timestep information into the model.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Tensor of shape (batch_size,) with timestep indices
        Returns:
            Tensor of shape (batch_size, dim) with sinusoidal embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        
        # Compute the frequency scale
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Create embeddings using sin and cos
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding conditioning.s
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels * 2),
            nn.GELU(),
        )
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Group normalization layers
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection projection if dimensions don't match
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tenor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            time_emb: Time embedding of shape (batch_size, time_emb_dim)
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # First block
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.conv1(h)
        
        # Add time embeddings using scale and shift
        time_scale_shift = self.time_mlp(time_emb)
        scale, shift = torch.chunk(time_scale_shift, 2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        # Second block
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip_conv(x)


class AttentionBlock(nn.Module):n
    """
    Multi-head self-attention block with group normalization.
    """
    def __init__(self, channels: int, num_heads: int = 4, num_head_channels: int = -1):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        if num_head_channels == -1:
            self.num_head_channels = channels // num_heads
        else:
            self.num_head_channels = num_head_channels
            self.num_heads = channels // self.num_head_channels
        
        # Group normalization
        self.norm = nn.GroupNorm(32, channels)
        
        # Linear projections for Q, K, V
        self.to_qkv = nn.Linear(channels, channels * 3)
        
        # Output projection
        self.to_out = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Output tensor of shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Reshape to (batch_size, height * width, channels)
        h = h.view(batch_size, channels, -1).transpose(1, 2)
        
        # Compute Q, K, V
        qkv = self.to_qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Reshape for multi-head attention
        seq_len = h.shape[1]
        q = q.view(batch_size, seq_len, self.num_heads, self.num_head_channels).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.num_head_channels).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.num_head_channels).transpose(1, 2)
        
        # Attention
        scale = self.num_head_channels ** -0.5
        sim = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(sim, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, channels)
        
        # Output projection
        out = self.to_out(out)
        
        # Reshape back to spatial dimensions
        out = out.transpose(1, 2).view(batch_size, channels, height, width)
        
        return out + x


class UNet(nn.Module):
    """
    U-Net architecture for denoising in diffusion models.
    Follows the architecture from the DDPM paper.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (8, 16),
        dropout: float = 0.1,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Encoder with downsampling
        self.down_blocks = nn.ModuleList()
        self.down_attention_blocks = nn.ModuleList()
        
        current_channels = model_channels
        resolution = 32  # Assuming 32x32 input, adjust as needed
        
        for level in range(3):  # 3 downsampling levels
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(current_channels, model_channels * (2 ** level), time_emb_dim, dropout)
                )
                current_channels = model_channels * (2 ** level)
                
                # Add attention if at specified resolution
                if resolution in attention_resolutions:
                    self.down_attention_blocks.append(AttentionBlock(current_channels))
                else:
                    self.down_attention_blocks.append(nn.Identity())
            
            # Downsample
            if level < 2:  # Don't downsample after the last level
                self.down_blocks.append(
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1)
                )
                self.down_attention_blocks.append(nn.Identity())
                resolution //= 2
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.middle_blocks.append(
                ResidualBlock(current_channels, current_channels, time_emb_dim, dropout)
            )
        
        # Add attention in middle
        self.middle_attention = AttentionBlock(current_channels)
        
        # Decoder with upsampling
        self.up_blocks = nn.ModuleList()
        self.up_attention_blocks = nn.ModuleList()
        
        resolution = 4  # Starting resolution after downsampling
        
        for level in range(2, -1, -1):  # 3 upsampling levels
            for _ in range(num_res_blocks + 1):
                skip_channels = model_channels * (2 ** level)
                self.up_blocks.append(
                    ResidualBlock(current_channels + skip_channels, model_channels * (2 ** level), time_emb_dim, dropout)
                )
                current_channels = model_channels * (2 ** level)
                
                # Add attention if at specified resolution
                if resolution in attention_resolutions:
                    self.up_attention_blocks.append(AttentionBlock(current_channels))
                else:
                    self.up_attention_blocks.append(nn.Identity())
            
            # Upsample
            if level > 0:
                self.up_blocks.append(
                    nn.Upsample(scale_factor=2, mode='nearest')store
                )
                self.up_attention_blocks.append(nn.Identity())
                resolution *= 2
        
        # Final output
        self.final_norm = nn.GroupNorm(32, model_channels)
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            timesteps: Timestep indices of shape (batch_size,)
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Embed timesteps
        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)
        
        # Initial convolution
        h = self.input_conv(x)
        
        # Store skip connections
        skips = [h]
        
        # Encoder
        down_idx = 0
        for i in range(len(self.down_blocks)):
            if isinstance(self.down_blocks[i], ResidualBlock):
                h = self.down_blocks[i](h, time_emb)
                h = self.down_attention_blocks[i](h)
                skips.append(h)
            else:  # Downsample
                h = self.down_blocks[i](h)
        
        # Middle
        for block in self.middle_blocks:
            h = block(h, time_emb)
        h = self.middle_attention(h)
        
        # Decoder
        for i in range(len(self.up_blocks)):
            if isinstance(self.up_blocks[i], ResidualBlock):
                # Concatenate skip connection
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[i](h, time_emb)
                h = self.up_attention_blocks[i](h)
            else:  # Upsample
                h = self.up_blocks[i](h)
        
        # Final output
        h = self.final_norm(h)
        h = F.gelu(h)
        output = self.final_conv(h)
        
        return output


class DiffusionModel(nn.Module):
    """
    Complete diffusion model with U-Net denoising network.
    Implements the forward and reverse processes from DDPM.
    """
    def __init__(
        self,
        image_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (8, 16),
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.image_channels = image_channels
        
        # U-Net denoising network
        self.unet = UNet(
            in_channels=image_channels,
            out_channels=image_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            time_emb_dim=model_channels,
        )
        
        # Register beta schedule as buffer
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            s = 0.008
            steps = torch.arange(num_timesteps + 1)
            alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * torch.tensor(math.pi) * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(max=0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Register buffers for the diffusion process
        self.register_buffer("betas", betas)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Variance schedule
        posterior_variance = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        
        # Precomputed coefficients for x_0 prediction
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
    
    def forward_diffusion(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> tuple:
        """
        Add noise to clean image according to diffusion process.
        
        Args:
            x0: Clean image tensor of shape (batch_size, channels, height, width)
            t: Timestep indices of shape (batch_size,)
            noise: Gaussian noise (if None, will be sampled)
        
        Returns:
            Tuple of (noisy_image, noise) where noisy_image is the corrupted image
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha.shape) < len(x0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        while len(sqrt_one_minus_alpha.shape) < len(x0.shape):
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        noisy_image = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        
        return noisy_image, noise
    
    def denoise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise at timestep t.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            t: Timestep indices of shape (batch_size,)
        
        Returns:
            Predicted noise of shape (batch_size, channels, height, width)
        """
        return self.unet(x, t)
    
    def backward_diffusion(
        self, x: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one step of reverse diffusion.
        
        Args:
            x: Current noisy image
            t: Current timestep indices
            predicted_noise: Predicted noise from model
        
        Returns:
            Image at previous timestep
        """
        beta_t = self.betas[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(beta_t.shape) < len(x.shape):
            beta_t = beta_t.unsqueeze(-1)
        while len(sqrt_one_minus_alpha.shape) < len(x.shape):
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        while len(sqrt_alpha.shape) < len(x.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        
        # x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
        x_prev = (x - (beta_t / sqrt_one_minus_alpha) * predicted_noise) / torch.sqrt(
            self.alphas[t].view(-1, 1, 1, 1)
        )
        
        return x_prev
    
    def sample(self, batch_size: int, image_shape: tuple = (3, 32, 32)) -> torch.Tensor:
        """
        Generate images by sampling from the reverse diffusion process.
        
        Args:
            batch_size: Number of images to generate
            image_shape: Shape of generated images (channels, height, width)
        
        Returns:
            Generated images of shape (batch_size, *image_shape)
        """
        device = next(self.parameters()).device
        
        # Start from pure noise
        x = torch.randn(batch_size, *image_shape, device=device)
        
        # Reverse diffusion
        for t in range(self.num_timesteps - 1, -1, -1):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.denoise(x, timesteps)
            
            # Go one step back
            if t > 0:
                noise = torch.randn_like(x)
                x = self.backward_diffusion(x, timesteps, predicted_noise)
                x = x + torch.sqrt(self.posterior_variance[t]) * noise
            else:
                x = self.backward_diffusion(x, timesteps, predicted_noise)
        
        return x


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = DiffusionModel(
        image_channels=3,
        model_channels=64,
        num_res_blocks=2,
        attention_resolutions=(8,),
        num_timesteps=1000,
    ).to(device)
    
    # Example: forward diffusion
    batch_size = 2
    x0 = torch.randn(batch_size, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    noisy_x, noise = model.forward_diffusion(x0, t)
    print(f"Input shape: {x0.shape}")
    print(f"Noisy image shape: {noisy_x.shape}")
    
    # Example: prediction
    predicted_noise = model.denoise(noisy_x, t)
    print(f"Predicted noise shape: {predicted_noise.shape}")
    
    # Example: sampling (slow without many iterations)
    # sample = model.sample(batch_size=2, image_shape=(3, 32, 32))
    # print(f"Generated image shape: {sample.shape}")
