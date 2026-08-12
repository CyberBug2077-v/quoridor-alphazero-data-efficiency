import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation module (Hu et al. 2017), Leela-style.
    Global-average-pool over the board to one scalar per channel, then
    FC (C -> C/r) -> ReLU -> FC (C/r -> C) -> sigmoid, used to rescale channels.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = x.mean(dim=(2, 3))                 # global average pool -> (B, C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))         # (B, C) in [0, 1]
        return x * s.view(B, C, 1, 1)          # channel-wise rescale


class ResBlock(nn.Module):
    """Standard AlphaZero residual block: Conv -> BN -> ReLU -> Conv -> BN, then skip.
    With se_enabled, a Squeeze-and-Excitation module rescales channels before the skip add."""

    def __init__(self, channels: int, se_enabled: bool = False, se_reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SEBlock(channels, se_reduction) if se_enabled else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return F.relu(out + residual, inplace=True)


class SelfAttention2D(nn.Module):
    """
    Single-layer multi-head self-attention over a spatial feature map.
    Operates on flattened (H*W) positions; output shape matches input.
    Used once after the residual tower to capture global wall-path interactions.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        # Learned positional embeddings for each spatial location
        # Board is always 17x17 = 289 positions
        self.pos_emb = nn.Parameter(torch.zeros(1, 289, channels))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dims: (B, H*W, C)
        seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        seq = seq + self.pos_emb[:, :H * W, :]
        normed = self.norm(seq)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        seq = seq + attn_out          # residual connection
        return seq.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

class QuoridorNNet(nn.Module):
    """
    AlphaZero-style architecture for Quoridor:
      - Input stem: 4-channel board -> `num_channels` feature maps
      - Body: `num_res_blocks` residual blocks (flat channels, no downsampling)
      - Global attention: 1 self-attention layer after the tower
      - Policy head: conv -> flatten -> linear -> action_size
      - Value  head: conv -> flatten -> linear -> tanh scalar
    """

    def __init__(self, game, args):
        super().__init__()

        self.board_x, self.board_y = game.getBoardSize()   # 17, 17
        self.action_size = game.getActionSize()            # 136

        C           = args.num_channels                    # 128
        num_res     = getattr(args, 'num_res_blocks', 6)
        attn_depth  = getattr(args, 'attn_depth', 1)       # 1 attention layer
        num_heads   = getattr(args, 'num_heads', 8)
        dropout     = getattr(args, 'dropout', 0.3)
        se_enabled  = bool(getattr(args, 'se_enabled', False))  # Squeeze-and-Excitation (off)
        # NHWC layout for the conv tower, used only under fast_opts (default off).
        self.channels_last = bool(getattr(args, 'fast_opts', False))

        # --- Stem ---
        self.stem = nn.Sequential(
            nn.Conv2d(4, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # --- Residual tower ---
        self.res_blocks = nn.Sequential(*[ResBlock(C, se_enabled=se_enabled) for _ in range(num_res)])

        # --- Global attention (1 layer) ---
        self.attn_layers = nn.ModuleList([
            SelfAttention2D(C, num_heads) for _ in range(attn_depth)
        ])

        # --- Policy head ---
        # Conv 2->32 channels, then flatten, linear to action_size
        self.policy_conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_drop = nn.Dropout(p=dropout)
        self.policy_fc   = nn.Linear(32 * self.board_x * self.board_y, self.action_size)

        # --- Value head ---
        # Conv 1->1 channel, then flatten, two linear layers, tanh
        self.value_conv = nn.Sequential(
            nn.Conv2d(C, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_drop = nn.Dropout(p=dropout)
        self.value_fc1  = nn.Linear(self.board_x * self.board_y, 256)
        self.value_fc2  = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s, logits: bool = False):
        # s: (batch, 4*H*W) or (batch, 4, H, W)
        x = s.view(-1, 4, self.board_x, self.board_y).float()

        # Optional NHWC layout so Tensor-Core conv kernels skip NCHW<->NHWC transposes.
        # Enabled only under fast_opts; default keeps the original contiguous (NCHW) layout
        # so behaviour is bit-for-bit unchanged. getattr guard keeps old checkpoints loading.
        if getattr(self, 'channels_last', False):
            x = x.to(memory_format=torch.channels_last)

        # Stem + residual tower
        x = self.stem(x)
        x = self.res_blocks(x)

        # Global attention
        for attn in self.attn_layers:
            x = attn(x)

        # Policy head
        pi = self.policy_conv(x)
        pi = self.policy_drop(pi.reshape(pi.size(0), -1))
        pi = self.policy_fc(pi)

        # Value head
        v = self.value_conv(x)
        v = self.value_drop(v.reshape(v.size(0), -1))
        v = F.relu(self.value_fc1(v), inplace=True)
        v = torch.tanh(self.value_fc2(v))

        if logits:
            return pi, v
        return F.softmax(pi, dim=1), v
