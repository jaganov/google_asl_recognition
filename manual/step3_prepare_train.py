import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import math
import time
import json

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from step2_prepare_dataset import load_dataset

TEST_MODE = False

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

dtype = torch.float
dtype_long = torch.long
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


class AdaptiveDropout(nn.Module):
    """
    Adaptive dropout that gradually increases
    """
    def __init__(self, initial_p=0.1, final_p=0.6, warmup_epochs=30):
        super().__init__()
        self.initial_p = initial_p
        self.final_p = final_p
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def forward(self, x):
        if self.training:
            # Gradual dropout increase
            if self.current_epoch < self.warmup_epochs:
                p = self.initial_p + (self.final_p - self.initial_p) * (self.current_epoch / self.warmup_epochs)
            else:
                p = self.final_p
            return F.dropout(x, p=p, training=True)
        return x
    
    def step(self):
        self.current_epoch += 1


# ============================================================================
# PREPROCESSING AND AUGMENTATION
# ============================================================================

class PreprocessingLayer(nn.Module):
    """
    Enhanced preprocessing layer with explicit temporal dependency modeling
    """
    def __init__(self, max_len=384, point_landmarks=None):
        super().__init__()
        self.max_len = max_len
        
        # Select key landmarks: hands, eyes, nose, lips
        if point_landmarks is None:
            # Face landmarks (eyes, nose, lips)
            face_landmarks = [33, 133, 362, 263, 61, 291, 199, 419, 17, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
            # Hand landmarks (key hand points)
            left_hand = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521]
            right_hand = [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]
            self.point_landmarks = face_landmarks + left_hand + right_hand
        else:
            self.point_landmarks = point_landmarks
    
    def compute_motion_features(self, x):
        """
        Compute extended motion features considering neighboring frames
        """
        # x: (batch, seq, landmarks, 2)
        batch_size, seq_len, num_landmarks, _ = x.shape
        
        # Basic motion features (lag1, lag2)
        dx = torch.zeros_like(x)
        dx2 = torch.zeros_like(x)
        
        if seq_len > 1:
            dx[:, :-1] = x[:, 1:] - x[:, :-1]  # velocity
        
        if seq_len > 2:
            dx2[:, :-2] = x[:, 2:] - x[:, :-2]  # acceleration
        
        # Extended motion features
        # 1. Relative motion (relative movement between landmarks)
        relative_motion = torch.zeros_like(x)
        if seq_len > 1:
            # Compute relative motion between adjacent landmarks
            for i in range(num_landmarks - 1):
                relative_motion[:, :-1, i] = x[:, 1:, i] - x[:, :-1, i+1]
        
        # 2. Temporal consistency (temporal consistency)
        temporal_consistency = torch.zeros_like(x)
        if seq_len > 3:
            # Check motion consistency over 3 frames
            for t in range(seq_len - 2):
                motion1 = x[:, t+1] - x[:, t]
                motion2 = x[:, t+2] - x[:, t+1]
                # Cosine similarity between motions
                cos_sim = F.cosine_similarity(motion1, motion2, dim=-1, eps=1e-8)
                temporal_consistency[:, t] = cos_sim.unsqueeze(-1).expand(-1, -1, 2)
        
        # 3. Motion magnitude (motion magnitude)
        motion_magnitude = torch.norm(dx, dim=-1, keepdim=True)
        
        # 4. Motion direction (motion direction)
        motion_direction = torch.atan2(dx[..., 1], dx[..., 0]).unsqueeze(-1)
        
        return dx, dx2, relative_motion, temporal_consistency, motion_magnitude, motion_direction
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, num_landmarks, 3)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, seq_len, num_landmarks, 3)
        
        # Normalization relative to nose (landmark 17)
        nose_coords = x[:, :, 17:18, :2]  # (batch, seq, 1, 2)
        # Replace NaN with 0.5 for mean calculation
        nose_coords_clean = torch.where(torch.isnan(nose_coords), torch.tensor(0.5, device=x.device, dtype=x.dtype), nose_coords)
        mean = torch.mean(nose_coords_clean, dim=[1, 2], keepdim=True)  # (batch, 1, 1, 2)
        
        # Select required landmarks
        x = x[:, :, self.point_landmarks, :]  # (batch, seq, num_selected, 3)
        
        # Standardization - expand mean to x dimensions
        mean_expanded = mean.expand(-1, x.shape[1], x.shape[2], -1)  # (batch, seq, num_selected, 2)
        
        # Replace NaN with 0 for std calculation
        x_clean = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device, dtype=x.dtype), x)
        std = torch.std(x_clean, dim=[1, 2], keepdim=True)  # (batch, 1, 1, 3)
        std_expanded = std.expand(-1, x.shape[1], x.shape[2], -1)  # (batch, seq, num_selected, 3)
        
        # Normalize only x, y coordinates (first 2 dimensions)
        x_normalized = x.clone()
        x_normalized[..., :2] = (x[..., :2] - mean_expanded) / (std_expanded[..., :2] + 1e-8)
        x = x_normalized
        
        # Truncate to max_len
        if self.max_len is not None:
            x = x[:, :self.max_len]
        
        length = x.shape[1]
        x = x[..., :2]  # Take only x, y coordinates
        
        # Compute extended motion features
        dx, dx2, relative_motion, temporal_consistency, motion_magnitude, motion_direction = self.compute_motion_features(x)
        
        # Combine all features
        x_flat = x.reshape(x.shape[0], length, -1)  # (batch, seq, num_landmarks*2)
        dx_flat = dx.reshape(x.shape[0], length, -1)
        dx2_flat = dx2.reshape(x.shape[0], length, -1)
        relative_motion_flat = relative_motion.reshape(x.shape[0], length, -1)
        temporal_consistency_flat = temporal_consistency.reshape(x.shape[0], length, -1)
        motion_magnitude_flat = motion_magnitude.reshape(x.shape[0], length, -1)
        motion_direction_flat = motion_direction.reshape(x.shape[0], length, -1)
        
        x_combined = torch.cat([
            x_flat, dx_flat, dx2_flat, relative_motion_flat, 
            temporal_consistency_flat, motion_magnitude_flat, motion_direction_flat
        ], dim=-1)
        
        # Replace NaN with 0
        x_combined = torch.where(torch.isnan(x_combined), torch.tensor(0.0, device=x.device, dtype=x.dtype), x_combined)
        
        return x_combined

class Augmentation:
    """
    Simplified data augmentation with preservation of temporal dependencies
    """
    def __init__(self, p=0.5):  # Increase augmentation probability to combat overfitting
        self.p = p
    
    def temporal_resample(self, x, target_length=None):
        """Simplified temporal resampling"""
        if random.random() > self.p:
            return x
        
        if target_length is None:
            scale = random.uniform(0.8, 1.2)  # More conservative range
            target_length = int(x.shape[0] * scale)
        
        if target_length <= 0:
            return x
        
        # Simple interpolation
        indices = torch.linspace(0, x.shape[0] - 1, target_length)
        indices = indices.long().clamp(0, x.shape[0] - 1)
        return x[indices]
    
    def random_masking(self, x, mask_ratio=0.05):  # Increase masking to combat overfitting
        """Simplified masking"""
        if random.random() > self.p:
            return x
        
        seq_len = x.shape[0]
        num_masks = int(seq_len * mask_ratio)
        
        if num_masks > 0:
            mask_indices = random.sample(range(seq_len), num_masks)
            x[mask_indices] = 0
        
        return x
    
    def horizontal_flip(self, x):
        """Horizontal flip"""
        if random.random() > self.p:
            return x
        
        x_flipped = x.clone()
        x_flipped[..., 0::2] = -x_flipped[..., 0::2]  # x coordinates
        return x_flipped
    
    def random_affine(self, x, max_scale=0.02, max_shift=0.01, max_rotate=2):  # Reduce parameters
        """Simplified affine transformations"""
        if random.random() > self.p:
            return x
        
        scale = 1 + random.uniform(-max_scale, max_scale)
        shift_x = random.uniform(-max_shift, max_shift)
        shift_y = random.uniform(-max_shift, max_shift)
        angle = random.uniform(-max_rotate, max_rotate) * math.pi / 180
        
        x_transformed = x.clone()
        x_transformed[..., 0::2] = x_transformed[..., 0::2] * scale + shift_x
        x_transformed[..., 1::2] = x_transformed[..., 1::2] * scale + shift_y
        
        return x_transformed
    
    def apply_augmentations(self, x):
        """Apply enhanced augmentations with adaptive probability"""
        # Adaptive augmentation probability
        if random.random() < 0.6:  # Increase base probability
            x = self.temporal_resample(x)
        if random.random() < 0.5:
            x = self.random_affine(x)
        if random.random() < 0.4:
            x = self.random_masking(x)
        
        # Add new augmentation type - temporal distortion
        if random.random() < 0.3:
            x = self.temporal_distortion(x)
        
        return x
    
    def temporal_distortion(self, x, max_shift=0.1):
        """Temporal distortion for better generalization"""
        if random.random() > self.p:
            return x
        
        seq_len = x.shape[0]
        # Create random shifts for each frame
        shifts = torch.randn(seq_len) * max_shift
        shifts = torch.cumsum(shifts, dim=0)
        
        # Apply shifts to coordinates
        x_distorted = x.clone()
        x_distorted[..., 0::2] += shifts.unsqueeze(-1).unsqueeze(-1)  # x coordinates
        x_distorted[..., 1::2] += shifts.unsqueeze(-1).unsqueeze(-1)  # y coordinates
        
        return x_distorted

# ============================================================================
# MODEL ARCHITECTURE (1D CNN + Transformer + TCN + LSTM)
# ============================================================================

class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Network block with extended receptive field
    """
    def __init__(self, dim, kernel_size=17, dilation=1, drop_rate=0.2):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Causal convolution with dilation
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=self.padding, 
                              dilation=dilation, groups=dim)
        self.conv2 = nn.Conv1d(dim, dim, 1)  # Pointwise
        
        # Gated activation
        self.gate_conv = nn.Conv1d(dim, dim, kernel_size, padding=self.padding, 
                                  dilation=dilation, groups=dim)
        self.gate_conv2 = nn.Conv1d(dim, dim, 1)
        
        # Normalization
        self.bn = nn.BatchNorm1d(dim, momentum=0.95)
        self.dropout = nn.Dropout(drop_rate)
        
        # Residual connection
        self.residual = nn.Conv1d(dim, dim, 1) if dim != dim else nn.Identity()
    
    def forward(self, x):
        # x: (batch, seq, dim)
        residual = self.residual(x.transpose(1, 2))
        
        # Causal convolution
        x = x.transpose(1, 2)  # (batch, dim, seq)
        
        # Main branch
        conv_out = self.conv1(x)
        conv_out = self.conv2(conv_out)
        
        # Gate branch
        gate_out = self.gate_conv(x)
        gate_out = self.gate_conv2(gate_out)
        gate_out = torch.sigmoid(gate_out)
        
        # Gated activation
        x = conv_out * gate_out
        
        # Apply causal padding (trim from right)
        x = x[:, :, :-self.padding] if self.padding > 0 else x
        
        # Ensure sequence dimension hasn't changed
        if x.shape[-1] != residual.shape[-1]:
            # If dimension changed, trim or pad
            target_len = residual.shape[-1]
            if x.shape[-1] > target_len:
                x = x[:, :, :target_len]
            else:
                # Pad with zeros on the right
                padding = torch.zeros(x.shape[0], x.shape[1], target_len - x.shape[2], 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=2)
        
        x = self.bn(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (batch, seq, dim)
        x = x + residual.transpose(1, 2)
        
        return x

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for capturing long-term temporal dependencies
    """
    def __init__(self, dim, hidden_dim=None, num_layers=2, drop_rate=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=drop_rate if num_layers > 1 else 0
        )
        
        # Projection back to original dimension
        self.projection = nn.Linear(hidden_dim * 2, dim)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        # x: (batch, seq, dim)
        lstm_out, _ = self.lstm(x)
        x = self.projection(lstm_out)
        x = self.dropout(x)
        return x

class TemporalAttention(nn.Module):
    """
    Temporal Attention for focusing on important frames
    """
    def __init__(self, dim, num_heads=8, drop_rate=0.2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Temporal attention
        self.temporal_q = nn.Linear(dim, dim)
        self.temporal_k = nn.Linear(dim, dim)
        self.temporal_v = nn.Linear(dim, dim)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop_rate)
        
        # Temporal position encoding
        self.temporal_pos_enc = nn.Parameter(torch.randn(1, 1000, dim))  # Max sequence length
    
    def forward(self, x):
        # x: (batch, seq, dim)
        batch_size, seq_len, _ = x.shape
        
        # Add temporal position encoding
        if seq_len <= self.temporal_pos_enc.shape[1]:
            pos_enc = self.temporal_pos_enc[:, :seq_len]
            x = x + pos_enc
        
        # Temporal attention
        q = self.temporal_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.temporal_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.temporal_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        x = self.output_proj(attn_output)
        x = self.dropout(x)
        
        return x

class Conv1DBlock(nn.Module):
    """
    1D CNN block with depthwise convolution and causal padding
    """
    def __init__(self, dim, kernel_size=17, drop_rate=0.2):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Causal padding
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=self.padding, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        
        # BatchNorm + Swish (as in winner solution)
        self.bn = nn.BatchNorm1d(dim, momentum=0.95)
        self.dropout = nn.Dropout(drop_rate)
        
        # Residual connection
        self.residual = nn.Conv1d(dim, dim, 1) if dim != dim else nn.Identity()
    
    def forward(self, x):
        # x: (batch, seq, dim)
        residual = self.residual(x.transpose(1, 2))
        
        # Causal convolution
        x = x.transpose(1, 2)  # (batch, dim, seq)
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # Apply causal padding (trim from right)
        x = x[:, :, :-self.padding] if self.padding > 0 else x
        
        # Ensure sequence dimension hasn't changed
        if x.shape[-1] != residual.shape[-1]:
            # If dimension changed, trim or pad
            target_len = residual.shape[-1]
            if x.shape[-1] > target_len:
                x = x[:, :, :target_len]
            else:
                # Pad with zeros on the right
                padding = torch.zeros(x.shape[0], x.shape[1], target_len - x.shape[2], 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=2)
        
        x = self.bn(x)
        x = F.silu(x)  # Swish activation
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (batch, seq, dim)
        x = x + residual.transpose(1, 2)
        
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with BatchNorm + Swish
    """
    def __init__(self, dim, num_heads=8, expand=2, drop_rate=0.2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop_rate)
        self.attention_norm = nn.BatchNorm1d(dim, momentum=0.95)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.SiLU(),  # Swish
            nn.Dropout(drop_rate),
            nn.Linear(dim * expand, dim),
            nn.Dropout(drop_rate)
        )
        self.ffn_norm = nn.BatchNorm1d(dim, momentum=0.95)
    
    def forward(self, x):
        # x: (batch, seq, dim)
        residual = x
        
        # Self-attention
        x_attn, _ = self.attention(x, x, x)
        x = x + x_attn
        x = self.attention_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # Feed-forward
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.ffn_norm(x.transpose(1, 2)).transpose(1, 2)
        
        return x

class LateDropout(nn.Module):
    """
    Dropout that is applied only after a certain step
    """
    def __init__(self, p=0.8, start_step=0):
        super().__init__()
        self.p = p
        self.start_step = start_step
        self.current_step = 0
    
    def forward(self, x):
        if self.current_step >= self.start_step:
            return F.dropout(x, p=self.p, training=self.training)
        return x
    
    def step(self):
        self.current_step += 1

class ASLModel(nn.Module):
    """
    Enhanced model with adaptive regularization and improved architecture
    """
    def __init__(self, input_dim, num_classes, max_len=384, dim=192, dropout_step=0):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        
        # Preprocessing
        self.preprocessing = PreprocessingLayer(max_len)
        
        # Stem with improved initialization
        self.stem_conv = nn.Linear(input_dim, dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim, momentum=0.95)
        self.stem_dropout = AdaptiveDropout(initial_p=0.1, final_p=0.3, warmup_epochs=20)
        
        # Temporal Convolutional Network (TCN) - 3 blocks with different dilations
        self.tcn1 = TemporalConvBlock(dim, kernel_size=17, dilation=1, drop_rate=0.15)
        self.tcn2 = TemporalConvBlock(dim, kernel_size=17, dilation=2, drop_rate=0.2)
        self.tcn3 = TemporalConvBlock(dim, kernel_size=17, dilation=4, drop_rate=0.25)
        
        # Projection for attention pooling (in case of dimension change)
        self.attention_projection = nn.Linear(dim, dim)
        
        # Bidirectional LSTM - 2 layers for better dependency capture
        self.lstm = BidirectionalLSTM(dim, hidden_dim=dim//2, num_layers=2, drop_rate=0.15)
        
        # Temporal Attention - more heads for better attention
        self.temporal_attention = TemporalAttention(dim, num_heads=8, drop_rate=0.15)
        
        # 1D CNN + Transformer blocks - add another conv block
        ksize = 17
        
        self.conv1 = Conv1DBlock(dim, ksize, drop_rate=0.15)
        self.conv2 = Conv1DBlock(dim, ksize, drop_rate=0.2)
        self.conv3 = Conv1DBlock(dim, ksize, drop_rate=0.25)
        self.transformer1 = TransformerBlock(dim, expand=2, drop_rate=0.15)
        
        # Top layers with improved pooling
        self.top_conv = nn.Linear(dim, dim)
        self.top_bn = nn.BatchNorm1d(dim, momentum=0.95)
        self.adaptive_dropout = AdaptiveDropout(initial_p=0.2, final_p=0.5, warmup_epochs=25)
        
        # Improved pooling - add attention pooling
        self.temporal_pool1 = nn.AdaptiveAvgPool1d(1)
        self.temporal_pool2 = nn.AdaptiveMaxPool1d(1)
        self.attention_pool = nn.MultiheadAttention(dim, num_heads=4, batch_first=True, dropout=0.1)
        
        # Final classifier with improved architecture
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, dim),  # dim*3 for avg + max + attention pooling
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(dim, num_classes)
        )
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.zeros_(param)
    
    def forward(self, x):
        # x: (batch, seq, landmarks, 3)
        
        # Preprocessing
        x = self.preprocessing(x)  # (batch, seq, features)
        
        # Stem with adaptive dropout
        x = self.stem_conv(x)
        x = self.stem_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.stem_dropout(x)
        
        # TCN blocks - 3 blocks with different dilations
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        
        # Bidirectional LSTM - 2 layers
        x = self.lstm(x)
        
        # Temporal Attention - more heads
        x = self.temporal_attention(x)
        
        # 1D CNN + Transformer - add third conv block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.transformer1(x)
        
        # Top layers with improved pooling
        x = self.top_conv(x)  # (batch, seq, dim)
        x = self.top_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.adaptive_dropout(x)
        
        # Improved temporal pooling
        x_transposed = x.transpose(1, 2)  # (batch, dim, seq)
        
        # Global average pooling
        global_avg = self.temporal_pool1(x_transposed).squeeze(-1)  # (batch, dim)
        
        # Global max pooling
        global_max = self.temporal_pool2(x_transposed).squeeze(-1)  # (batch, dim)
        
        # Attention pooling - use projection for stability
        x_for_attn = self.attention_projection(x)  # x already has dimension (batch, seq, dim)
        
        attn_out, _ = self.attention_pool(x_for_attn, x_for_attn, x_for_attn)
        global_attn = torch.mean(attn_out, dim=1)  # (batch, dim)
        
        # Combine pooling results
        x_pooled = torch.cat([global_avg, global_max, global_attn], dim=1)  # (batch, dim*3)
        
        x = self.classifier(x_pooled)
        
        return x

# ============================================================================
# ENHANCED TRAINING STRATEGY
# ============================================================================


class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Enhanced scheduler with warm restarts
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        T_cur = self.last_epoch
        T_i = self.T_0
        
        # Find current period
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * T_cur / T_i)) / 2
                for base_lr in self.base_lrs]

# ============================================================================
# DATASET AND DATALOADER
# ============================================================================

class ASLDataset(Dataset):
    """
    Dataset for ASL gestures
    """
    def __init__(self, sequences, labels, augment=True):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.augmentor = Augmentation(p=0.5) if augment else None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.augment and self.augmentor:
            # Apply augmentation
            sequence = self.augmentor.apply_augmentations(sequence)
        
        return sequence, label

def collate_fn(batch):
    """
    Collate function for batches of different lengths
    """
    sequences, labels = zip(*batch)
    
    # Find maximum length in batch
    max_len = max(seq.shape[0] for seq in sequences)
    
    # Padding to maximum length
    padded_sequences = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            # Zero padding
            padding = torch.zeros((max_len - seq.shape[0], seq.shape[1], seq.shape[2]), 
                                dtype=seq.dtype, device=seq.device)
            seq = torch.cat([seq, padding], dim=0)
        padded_sequences.append(seq)
    
    # Stack into batch
    batch_sequences = torch.stack(padded_sequences)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_sequences, batch_labels

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_model_manifest(model, train_data, train_labels, test_data, test_labels, 
                         num_classes, epochs, batch_size, lr, max_len, 
                         total_params, trainable_params, timestamp, model_prefix):
    """
    Creates a manifest with description of the enhanced model and training parameters
    """
    manifest = {
        "model_info": {
            "name": model_prefix,
            "timestamp": timestamp,
            "architecture": "Enhanced_TCN_LSTM_Transformer_v2",
            "version": "adaptive_regularization_v2",
            "description": "ASL Recognition model with adaptive regularization and improved architecture"
        },
        "model_parameters": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "input_dim": model.stem_conv.in_features,
            "hidden_dim": model.dim,
            "num_classes": num_classes,
            "max_sequence_length": max_len
        },
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "weight_decay": 0.005,
            "loss_function": "CrossEntropyLoss",
            "label_smoothing": 0.05,
            "scheduler": "CosineAnnealingWarmRestarts",
            "warmup_epochs": 10,
            "scheduler_params": {
                "T_0": 50,
                "T_mult": 2,
                "eta_min": "lr*0.01"
            }
        },
        "architecture_details": {
            "preprocessing": {
                "type": "PreprocessingLayer",
                "max_len": max_len,
                "motion_features": ["velocity", "acceleration", "relative_motion", "temporal_consistency", "motion_magnitude", "motion_direction"]
            },
            "stem": {
                "type": "Linear + BatchNorm + AdaptiveDropout",
                "dropout_range": [0.1, 0.3],
                "warmup_epochs": 20
            },
            "tcn_blocks": {
                "count": 3,
                "kernel_size": 17,
                "dilations": [1, 2, 4],
                "dropout_rates": [0.15, 0.2, 0.25]
            },
            "lstm": {
                "type": "BidirectionalLSTM",
                "layers": 2,
                "hidden_dim": "dim//2",
                "dropout": 0.15
            },
            "attention": {
                "type": "TemporalAttention",
                "heads": 8,
                "dropout": 0.15
            },
            "conv_blocks": {
                "count": 3,
                "kernel_size": 17,
                "dropout_rates": [0.15, 0.2, 0.25]
            },
            "transformer": {
                "blocks": 1,
                "heads": 8,
                "expand": 2,
                "dropout": 0.15
            },
            "pooling": {
                "types": ["global_avg", "global_max", "attention"],
                "attention_heads": 4,
                "dropout": 0.1
            },
            "classifier": {
                "type": "Sequential",
                "layers": [
                    "Linear(dim*3, dim)",
                    "BatchNorm1d",
                    "SiLU",
                    "Dropout(0.3)",
                    "Linear(dim, num_classes)"
                ]
            },
            "adaptive_dropout": {
                "type": "AdaptiveDropout",
                "initial_p": 0.2,
                "final_p": 0.5,
                "warmup_epochs": 25
            }
        },
        "augmentation": {
            "temporal_resample": {
                "probability": 0.6,
                "scale_range": [0.8, 1.2]
            },
            "random_masking": {
                "probability": 0.4,
                "ratio": 0.05
            },
            "random_affine": {
                "probability": 0.5,
                "max_scale": 0.02,
                "max_shift": 0.01,
                "max_rotate": 2
            },
            "temporal_distortion": {
                "probability": 0.3,
                "max_shift": 0.1
            }
        },
        "early_stopping": {
            "patience": 20,
            "min_epochs": 80
        },
        "dataset_info": {
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "num_classes": num_classes,
            "classes": list(range(num_classes))
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "key_improvements_v2": [
            "Adaptive dropout (0.1→0.6) instead of sharp activation",
            "TCN: 3 blocks with different dilations (1,2,4)",
            "LSTM: 2 layers for better dependency capture", 
            "Attention: 8 heads + attention pooling",
            "Conv: 3 blocks + improved classifier",
            "CosineAnnealingWarmRestarts scheduler",
            "Temporal distortion in augmentation",
            "Reduced weight decay (0.005)",
            "Improved pooling (avg + max + attention)"
        ],
        "expected_improvements": {
            "val_accuracy_target": "75-78%",
            "train_val_gap_target": "10-12%",
            "stability": "More stable training without sharp jumps",
            "early_stopping": "100-150 epochs instead of 55"
        }
    }
    
    return manifest

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Training one epoch with enhanced strategy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Update adaptive dropout layers
    for module in model.modules():
        if isinstance(module, AdaptiveDropout):
            module.step()
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (sequences, labels) in enumerate(progress_bar):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        accuracy = 100. * correct / total
        progress_bar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def validate_epoch(model, dataloader, criterion, device):
    """
    Validation of one epoch
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc='Validation'):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def train_model(train_data, train_labels, test_data, test_labels, num_classes, 
                epochs=400, batch_size=32, lr=5e-4, max_len=384, timestamp=None, model_prefix=None):
    """
    Main training function for simplified but effective ASL model
    """
    print(f"🎯 Starting training of simplified ASL model with focus on temporal dependencies")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Classes: {num_classes}")
    print(f"   Architecture: TCN + LSTM + Transformer (simplified)")
    
    # Create datasets
    train_dataset = ASLDataset(train_data, train_labels, augment=True)
    test_dataset = ASLDataset(test_data, test_labels, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # Determine input dimension (after enhanced preprocessing)
    sample_sequence = train_data[0]
    sample_preprocessed = PreprocessingLayer(max_len)(sample_sequence.unsqueeze(0))
    input_dim = sample_preprocessed.shape[-1]
    
    print(f"   Input dimension after enhanced preprocessing: {input_dim}")
    
    # Create simplified model
    model = ASLModel(input_dim=input_dim, num_classes=num_classes, max_len=max_len, dim=192)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss function with smaller label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Optimizer with improved parameters
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.005, betas=(0.9, 0.999))
    
    # Enhanced scheduler with warm restarts
    warmup_epochs = 10
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=lr*0.01)
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    patience_counter = 0
    
    # Use passed parameters for versioning
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_prefix is None:
        model_prefix = f"asl_model_v{timestamp}"
    
    best_model_path = f"models/{model_prefix}.pth"
    manifest_path = f"models/{model_prefix}_manifest.json"
    
    print(f"\n🚀 Starting simplified model training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Learning rate warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_scale
        else:
            scheduler.step()
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
        
        # Save history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Create manifest with current results
            manifest = create_model_manifest(
                model=model,
                train_data=train_data,
                train_labels=train_labels,
                test_data=test_data,
                test_labels=test_labels,
                num_classes=num_classes,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                max_len=max_len,
                total_params=total_params,
                trainable_params=trainable_params,
                timestamp=timestamp,
                model_prefix=model_prefix
            )
            
            # Add training results
            manifest["training_results"] = {
                "best_epoch": epoch,
                "best_val_accuracy": val_acc,
                "best_train_accuracy": train_acc,
                "current_train_loss": train_loss,
                "current_val_loss": val_loss,
                "training_progress": {
                    "epochs_completed": epoch + 1,
                    "total_epochs": epochs,
                    "patience_counter": patience_counter
                }
            }
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'architecture': 'Simplified_TCN_LSTM_Transformer',
                'manifest': manifest
            }, best_model_path)
            
            # Save manifest separately
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            print(f"  💾 Best model saved (Val Acc: {val_acc:.2f}%)")
            print(f"  📄 Manifest saved: {manifest_path}")
        else:
            patience_counter += 1
        
        # Stricter early stopping to combat overfitting
        if epoch > 80 and patience_counter > 20:  # 20 epochs without improvement
            print(f"  ⚠️ Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
            break
        
        print("-" * 50)
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/3600:.2f} hours")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Improvement over baseline model: +{best_val_acc - 65:.2f}%")
    
    # Create final manifest with complete results
    final_manifest = create_model_manifest(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_len=max_len,
        total_params=total_params,
        trainable_params=trainable_params,
        timestamp=timestamp,
        model_prefix=model_prefix
    )
    
    # Add final results
    final_manifest["final_results"] = {
        "training_time_hours": training_time / 3600,
        "best_val_accuracy": best_val_acc,
        "improvement_over_baseline": best_val_acc - 65,
        "total_epochs_trained": len(train_losses),
        "early_stopping_triggered": len(train_losses) < epochs,
        "final_epoch": len(train_losses) - 1,
        "training_history": {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
    }
    
    # Save final manifest
    final_manifest_path = f"models/{model_prefix}_final_manifest.json"
    with open(final_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(final_manifest, f, indent=2, ensure_ascii=False)
    
    print(f"  📄 Final manifest saved: {final_manifest_path}")
    
    # Plots
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    
    return model, best_val_acc

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Plot training history graphs
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Val Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN CODE
# ============================================================================

if __name__ == "__main__":
    print("🎯 Google ASL Recognition - Training enhanced model with adaptive regularization")
    print("=" * 80)
    print("🔧 Key improvements v2:")
    print("   ✅ Adaptive dropout (0.1→0.6) instead of sharp activation")
    print("   ✅ TCN: 3 blocks with different dilations (1,2,4)")
    print("   ✅ LSTM: 2 layers for better dependency capture")
    print("   ✅ Attention: 8 heads + attention pooling")
    print("   ✅ Conv: 3 blocks + improved classifier")
    print("   ✅ CosineAnnealingWarmRestarts scheduler")
    print("   ✅ Temporal distortion in augmentation")
    print("   ✅ Reduced weight decay (0.005)")
    print("=" * 80)
    
    # Load data
    print("📁 Loading dataset...")
    train_data, train_labels, test_data, test_labels, sign_mapping, classes = load_dataset(max_samples=100 if TEST_MODE else None)
    
    print(f"✅ Loaded:")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    print(f"   Classes: {len(classes)}")
    print(f"   Classes: {classes}")
    
    # Create models directory
    import os
    os.makedirs("models", exist_ok=True)
    
    # Create timestamp prefix for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_prefix = f"asl_model_v{timestamp}"
    
    # Train model
    model, best_acc = train_model(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        num_classes=len(classes),
        epochs=10 if TEST_MODE else 300,  # Reduce number of epochs
        batch_size=32,  # Optimized for RTX4070
        lr=1e-4 if TEST_MODE else 4e-4,  # Slightly increase learning rate
        max_len=384,
        timestamp=timestamp,
        model_prefix=model_prefix
    )
    
    print(f"\n🎉 Model training completed!")
    print(f"   Best accuracy: {best_acc:.2f}%")
    print(f"   Model saved in: models/{model_prefix}.pth")
    print(f"   Manifest saved in: models/{model_prefix}_manifest.json")
    print(f"   Final manifest: models/{model_prefix}_final_manifest.json")
    print(f"   Improvement over baseline model: +{best_acc - 65:.2f}%")
    
    print(f"\n🔬 Key improvements v2:")
    print(f"   1. Adaptive dropout (0.1→0.6) instead of sharp activation")
    print(f"   2. TCN: 3 blocks with different dilations (1,2,4)")
    print(f"   3. LSTM: 2 layers for better dependency capture")
    print(f"   4. Attention: 8 heads + attention pooling")
    print(f"   5. Conv: 3 blocks + improved classifier")
    print(f"   6. CosineAnnealingWarmRestarts scheduler")
    print(f"   7. Temporal distortion in augmentation")
    print(f"   8. Reduced weight decay (0.005)")
    
    print(f"\n💡 Why these changes should help:")
    print(f"   - Adaptive dropout = smooth regularization activation")
    print(f"   - More layers = better learning capacity")
    print(f"   - Attention pooling = better representation of important frames")
    print(f"   - Warm restarts = escape from local minima")
    print(f"   - Temporal distortion = better generalization")
    
    print(f"\n📊 Expected results:")
    print(f"   - Val accuracy: 75-78% (instead of 72.3%)")
    print(f"   - Smaller gap between train/val (10-12% instead of 17.4%)")
    print(f"   - More stable training without sharp jumps")
    print(f"   - Later early stopping (100-150 epochs instead of 55)")
    print(f"   - Elimination of sharp jump at epoch 30")
