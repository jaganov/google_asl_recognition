# 🏋️ Training Guide

This guide covers the complete training process for the ASL recognition model using `step3_prepare_train.py`.

## 🎯 Overview

The training script implements a state-of-the-art ASL recognition model with:
- **Hybrid Architecture**: TCN + LSTM + Transformer blocks
- **Adaptive Regularization**: Advanced dropout and regularization techniques
- **RTX4070 Optimizations**: GPU-specific performance enhancements
- **Comprehensive Monitoring**: Training history and manifest system

## 🏗️ Model Architecture

### Core Components

#### 1. Preprocessing Layer
```python
class PreprocessingLayer(nn.Module):
    """
    Enhanced preprocessing with motion features and normalization
    """
    def __init__(self, max_len=384, point_landmarks=None):
        # Select key landmarks: hands, eyes, nose, lips
        # Compute motion features: velocity, acceleration, relative motion
        # Normalize relative to nose landmark
```

**Features**:
- **Landmark Selection**: 62 key points (face + hands)
- **Motion Features**: Velocity, acceleration, relative motion, temporal consistency
- **Normalization**: Relative to nose landmark for consistency

#### 2. TCN Blocks
```python
class TemporalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=17, dilation=1, drop_rate=0.2):
        # 1D convolution with dilation
        # BatchNorm + ReLU + Dropout
        # Residual connection
```

**Configuration**:
- **3 TCN blocks** with dilations [1, 2, 4]
- **Kernel size**: 17 for temporal context
- **Dropout rates**: [0.15, 0.2, 0.25] (increasing)

#### 3. Bidirectional LSTM
```python
class BidirectionalLSTM(nn.Module):
    def __init__(self, dim, hidden_dim=None, num_layers=2, drop_rate=0.2):
        # 2-layer bidirectional LSTM
        # Hidden dim = dim//2 for each direction
        # Dropout between layers
```

#### 4. Temporal Attention
```python
class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, drop_rate=0.2):
        # Multi-head self-attention
        # 8 attention heads
        # Dropout for regularization
```

#### 5. Conv1D + Transformer Blocks
```python
class Conv1DBlock(nn.Module):
    # 1D convolution with residual connection

class TransformerBlock(nn.Module):
    # Transformer block with self-attention and FFN
```

## 🔧 Training Configuration

### Hyperparameters
```python
# Model Parameters
input_dim = 744          # 62 landmarks × 3 coordinates × 4 features
hidden_dim = 192         # Model dimension
num_classes = 25         # ASL gestures
max_len = 384           # Maximum sequence length

# Training Parameters
epochs = 300            # Total training epochs
batch_size = 32         # Optimized for RTX4070
learning_rate = 4e-4    # AdamW learning rate
weight_decay = 0.005    # L2 regularization
```

### Optimizer & Scheduler
```python
# AdamW Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.999)
)

# Cosine Annealing with Warm Restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,             # Initial restart period
    T_mult=2,           # Period multiplier
    eta_min=lr*0.01     # Minimum learning rate
)
```

### Loss Function
```python
# CrossEntropyLoss with Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
```

## 🛡️ Adaptive Regularization

### 1. Adaptive Dropout
```python
class AdaptiveDropout(nn.Module):
    def __init__(self, initial_p=0.1, final_p=0.6, warmup_epochs=30):
        # Gradual dropout increase during training
        # Prevents overfitting while maintaining learning capacity
```

**Behavior**:
- **Initial**: 10% dropout
- **Warmup**: Gradual increase over 30 epochs
- **Final**: 60% dropout

### 2. Late Dropout
```python
class LateDropout(nn.Module):
    def __init__(self, p=0.8, start_step=0):
        # High dropout applied later in training
        # Helps with generalization
```

### 3. DropPath (Stochastic Depth)
```python
# Applied in TCN and Conv1D blocks
# Randomly drops entire blocks during training
```

## 🚀 RTX4070 Optimizations

### 1. Mixed Precision Training
```python
# Automatic mixed precision (FP16)
# Reduces memory usage and speeds up training
# Compatible with RTX4070 Tensor Cores
```

### 2. Memory Optimizations
```python
# Automatic batch size tuning
# Gradient accumulation for large effective batch sizes
# Memory-efficient data loading
```

### 3. cuDNN Benchmark
```python
# Optimized convolution algorithms
# Faster training on RTX4070
```

## 📊 Training Process

### 1. Data Loading
```python
# Load preprocessed dataset
train_data, train_labels, test_data, test_labels, sign_mapping, classes = load_dataset(
    data_dir="dataset25_split",
    max_len=384,
    max_samples=None  # Use all samples
)
```

### 2. Model Initialization
```python
# Create model with adaptive regularization
model = ASLModel(
    input_dim=744,
    num_classes=25,
    max_len=384,
    dim=192,
    dropout_step=0
)
```

### 3. Training Loop
```python
for epoch in range(epochs):
    # Training phase
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    # Validation phase
    val_loss, val_acc = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # Learning rate scheduling
    scheduler.step()
    
    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Save best model
```

### 4. Early Stopping
```python
# Stop training if validation accuracy doesn't improve
# Patience: 20 epochs
# Save best model automatically
```

## 📈 Training Monitoring

### 1. Real-time Metrics
- **Training Loss**: Cross-entropy loss
- **Training Accuracy**: Percentage of correct predictions
- **Validation Loss**: Loss on validation set
- **Validation Accuracy**: Accuracy on validation set
- **Learning Rate**: Current learning rate

### 2. Training History
```python
# Track all metrics during training
training_history = {
    'train_losses': [...],
    'train_accuracies': [...],
    'val_losses': [...],
    'val_accuracies': [...],
    'learning_rates': [...]
}
```

### 3. Manifest System
```python
# Create comprehensive training manifest
manifest = {
    'model_info': {...},
    'training_config': {...},
    'architecture_details': {...},
    'training_history': {...},
    'final_results': {...}
}
```

## 🎯 Expected Results

### Training Timeline
- **Total Epochs**: 300 (full training)
- **Early Stopping**: Usually around 80-100 epochs
- **Training Time**: ~1.75 hours on RTX4070
- **Memory Usage**: ~8-10GB GPU memory

### Performance Metrics
- **Best Validation Accuracy**: ~76.05%
- **Training Accuracy**: ~85-90%
- **Model Parameters**: 1.97M
- **Inference Time**: <50ms per prediction

## 🔧 Usage

### Basic Training
```bash
cd manual
python step3_prepare_train.py
```

### Training with Custom Parameters
```python
# Modify parameters in step3_prepare_train.py
TEST_MODE = False  # Set to True for quick testing
epochs = 300       # Adjust training epochs
batch_size = 32    # Adjust batch size
learning_rate = 4e-4  # Adjust learning rate
```

### Monitoring Training
```python
# Training progress is displayed in real-time
# Checkpoints are saved automatically
# Final model and manifest are saved in models/
```

## 📊 Model Outputs

### Saved Files
1. **Model Weights**: `asl_model_vYYYYMMDD_HHMMSS.pth`
2. **Training Manifest**: `asl_model_vYYYYMMDD_HHMMSS_manifest.json`
3. **Final Manifest**: `asl_model_vYYYYMMDD_HHMMSS_final_manifest.json`
4. **Training Plot**: `training_history.png`

### Manifest Contents
- **Model Architecture**: Complete model configuration
- **Training Parameters**: All hyperparameters and settings
- **Training History**: Loss and accuracy curves
- **Performance Metrics**: Final results and analysis
- **Hardware Info**: GPU specifications and optimizations

## 🚨 Troubleshooting

### Common Issues

#### 1. Out of Memory
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Enable gradient accumulation
gradient_accumulation_steps = 2
```

#### 2. Slow Training
```python
# Check GPU utilization
nvidia-smi

# Enable mixed precision
torch.backends.cudnn.benchmark = True
```

#### 3. Poor Accuracy
```python
# Increase regularization
dropout_rate += 0.1

# Adjust learning rate
learning_rate *= 0.5
```

## 📚 Related Documentation

- **[Model Architecture](architecture.md)** - Detailed architecture explanation
- **[RTX4070 Optimizations](rtx4070-optimizations.md)** - GPU-specific optimizations
- **[Manifest System](manifest-system.md)** - Model versioning and tracking
- **[Model Results](model-results.md)** - Performance analysis

---

**Ready to train your ASL recognition model! 🚀** 