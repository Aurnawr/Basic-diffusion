import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    #intialize the constructor
    def __init__(self, dim : int):
        super().__init__()
        self.dim= dim
    
    def forward (self, time: torch.Tensor) -> torch.Tensor :
        # time -> tensor of timestep indices of size (batch_size, )
        device = time.device
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        # returns ( batch_size, dim)

        return emb

class ResidualBlock(nn.Module):
    #initialize the channels and the dimention
    def __init__(self, in_channels : int, out_channels: int, time_emb_dim: int, dropout: float= 0.1):
        super().__init__()
        self.in_ch= in_channels
        self.out_ch= out_channels
        self.time_emb_dim= time_emb_dim

        #first conv layer
        self.conv1= nn.Conv2d(in_channels,out_channels, kernel_size=3, padding= 1)

         # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels * 2),
            nn.GELU(),
        )

        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        #Group norm layers
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if dimensions don't match
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x : torch.Tensor,time_emb: torch.Tensor ) -> torch.Tensor:
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

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int= 4, num_head_channels= -1 ):
        super().__init__()


        self.channels= channels
        self.num_heads = num_heads

        #number of channels per head 
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
    
class UNet (nn.Module):
    def __init__ ( 
            self, 
            in_channels= 3,
            out_channels= 3,
            model_channels=128,
            num_res_blocks= 2,
            attention_resolutions= (8,16),
            dropout = 0.1,
            time_emb_dim = 128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout= dropout
        self.model_channels= model_channels
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions= attention_resolutions

        # Time embedding
        self.time_embedding = SinusoidalPositionalEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Encoder with downsampling
        self.down_blocks = nn.ModuleList()
        self.down_attention_blocks = nn.ModuleList() # a list of modules (layers)
        
        current_channels = model_channels
        resolution = 32  

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
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
                self.up_attention_blocks.append(nn.Identity())
                resolution *= 2
        
        # Final output
        self.final_norm = nn.GroupNorm(32, model_channels)
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps:torch.Tensor ):
        




        


