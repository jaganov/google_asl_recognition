# 🤟 Google ASL Signs Recognition Project
## Complete 1-Week Development Roadmap

> **Hardware:** NVIDIA RTX 4070 12GB + 32GB RAM + Alienware Aurora  
> **Dataset:** Google ASL Signs (250 ASL words)  
> **Goal:** 80-85% accuracy with real-time webcam demo  
> **Timeline:** 7 days intensive development  

---

## 📋 Project Overview

### What We're Building
- **Model:** Vision Transformer (ViT) + CNN ensemble for ASL recognition
- **Dataset:** Google ASL Signs - 250 words, ~100k video samples
- **Output:** Real-time webcam demo + comprehensive metrics
- **Tech Stack:** PyTorch, Transformers, OpenCV, Gradio

### Expected Results
- **Training Accuracy:** 90-95%
- **Validation Accuracy:** 80-85%
- **Inference Time:** <50ms per prediction on RTX 4070
- **Model Size:** ~200-300MB

---

## 🗂️ Project Structure
```
google_asl_recognition/
├── data/
│   ├── google_asl_signs/        # Downloaded dataset
│   ├── processed/               # Preprocessed data
│   └── splits/                  # train/val/test splits
├── models/
│   ├── vision_transformer/      # ViT architecture
│   ├── cnn_backbone/           # ResNet/EfficientNet
│   ├── ensemble/               # Combined models
│   └── weights/                # Saved checkpoints
├── src/
│   ├── data_loader.py          # Data loading utilities
│   ├── train_vit.py            # ViT training script
│   ├── train_cnn.py            # CNN training script
│   ├── ensemble_train.py       # Ensemble training
│   ├── inference.py            # Real-time inference
│   └── utils/                  # Helper functions
├── notebooks/                   # Experiments & analysis
├── demo/                       # Web demo application
├── configs/                    # Model configurations
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## 📅 7-Day Intensive Timeline

### Day 1: Setup & Data Preparation (Monday)
**Morning (4 hours):**
- [ ] Environment setup with CUDA 11.8+
- [ ] Download Google ASL Signs dataset
- [ ] Explore dataset structure and statistics
- [ ] Set up project repository

**Afternoon (4 hours):**
- [ ] Implement data preprocessing pipeline
- [ ] Create train/validation/test splits (80/10/10)
- [ ] Verify data loading and augmentation
- [ ] Baseline data analysis

### Day 2: CNN Baseline Implementation (Tuesday)
**Morning (4 hours):**
- [ ] Implement ResNet50 + 3D CNN baseline
- [ ] Set up training loop with monitoring
- [ ] Configure loss functions and optimizers
- [ ] Start baseline training

**Afternoon (4 hours):**
- [ ] Monitor training progress
- [ ] Implement early stopping and checkpointing
- [ ] Hyperparameter tuning
- [ ] Target: 70-75% validation accuracy

### Day 3: Vision Transformer Implementation (Wednesday)
**Morning (4 hours):**
- [ ] Implement Video Vision Transformer (ViViT)
- [ ] Adapt ViT for video sequence processing
- [ ] Configure temporal attention mechanisms
- [ ] Set up ViT training pipeline

**Afternoon (4 hours):**
- [ ] Start ViT training with pretrained weights
- [ ] Compare ViT vs CNN performance
- [ ] Optimize ViT hyperparameters
- [ ] Target: 75-80% validation accuracy

### Day 4: Model Optimization & Ensemble (Thursday)
**Morning (4 hours):**
- [ ] Implement ensemble of CNN + ViT
- [ ] Advanced data augmentation strategies
- [ ] Learning rate scheduling optimization
- [ ] Mixed precision training

**Afternoon (4 hours):**
- [ ] Ensemble training and validation
- [ ] Model compression techniques
- [ ] Inference optimization
- [ ] Target: 80-85% validation accuracy

### Day 5: Real-time Demo Development (Friday)
**Morning (4 hours):**
- [ ] Implement real-time webcam inference
- [ ] Build Gradio web interface
- [ ] Optimize inference pipeline for speed
- [ ] Add confidence thresholding

**Afternoon (4 hours):**
- [ ] Polish demo interface
- [ ] Add prediction visualization
- [ ] Implement top-K predictions display
- [ ] Performance monitoring dashboard

### Day 6: Testing & Refinement (Saturday)
**Morning (4 hours):**
- [ ] Comprehensive model evaluation
- [ ] Error analysis and confusion matrices
- [ ] Demo stress testing
- [ ] Performance benchmarking

**Afternoon (4 hours):**
- [ ] Final model fine-tuning
- [ ] Demo UI improvements
- [ ] Documentation writing
- [ ] Prepare presentation materials

### Day 7: Final Polish & Deployment (Sunday)
**Morning (3 hours):**
- [ ] Final testing and bug fixes
- [ ] Model export for deployment
- [ ] Create demo video
- [ ] Performance report generation

**Afternoon (2 hours):**
- [ ] Final documentation
- [ ] Code cleanup and commenting
- [ ] Repository organization
- [ ] Project completion celebration! 🎉

---

## 🛠️ Technical Implementation

### Dataset: Google ASL Signs
```python
# Dataset specifications
DATASET_INFO = {
    'name': 'Google ASL Signs',
    'words': 250,
    'total_videos': ~100000,
    'video_length': '1-3 seconds',
    'resolution': '512x512 or variable',
    'format': 'MP4',
    'splits': 'Pre-defined train/val/test',
    'download_size': '~50-100GB'
}

# Download command
# kaggle datasets download -d google/asl-signs
# or use TensorFlow Datasets
import tensorflow_datasets as tfds
ds = tfds.load('asl_signs', split='train', as_supervised=True)
```

### Model Architecture 1: Enhanced CNN
```python
class ASLCNNModel(nn.Module):
    def __init__(self, num_classes=250):
        super().__init__()
        # EfficientNet-B3 backbone (better than ResNet50)
        self.backbone = timm.create_model(
            'efficientnet_b3', 
            pretrained=True, 
            num_classes=0,  # Remove head
            global_pool=''
        )
        
        # 3D CNN for temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(1536, 512, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        batch_size, frames = x.shape[:2]
        
        # Process each frame through backbone
        x = x.view(-1, *x.shape[2:])  # Flatten batch and frames
        features = self.backbone(x)   # Extract features
        
        # Reshape for 3D conv
        features = features.view(batch_size, frames, *features.shape[1:])
        features = features.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # Temporal modeling
        temporal_features = self.temporal_conv(features)
        temporal_features = temporal_features.view(batch_size, -1)
        
        # Classification
        output = self.classifier(temporal_features)
        return output
```

### Model Architecture 2: Video Vision Transformer
```python
class VideoViT(nn.Module):
    def __init__(self, num_classes=250, num_frames=16):
        super().__init__()
        # Use pretrained ViT and adapt for video
        self.patch_embed = PatchEmbed3D(
            img_size=224, 
            patch_size=16, 
            num_frames=num_frames,
            in_chans=3, 
            embed_dim=768
        )
        
        # Temporal and spatial positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, 768)
        )
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, num_frames, 768)
        )
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=12
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        B, T, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Add temporal embeddings (broadcast across patches)
        temporal_emb = self.temporal_embed.repeat_interleave(
            x.shape[1] // T, dim=1
        )
        x = x + temporal_emb
        
        # Transformer processing
        x = self.transformer(x)
        
        # Global average pooling and classification
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        
        return x
```

### Ensemble Model
```python
class ASLEnsemble(nn.Module):
    def __init__(self, cnn_model, vit_model, num_classes=250):
        super().__init__()
        self.cnn = cnn_model
        self.vit = vit_model
        
        # Ensemble fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Learnable weights for ensemble
        self.cnn_weight = nn.Parameter(torch.tensor(0.5))
        self.vit_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        # Get predictions from both models
        cnn_out = self.cnn(x)
        vit_out = self.vit(x)
        
        # Weighted ensemble
        ensemble_logits = (
            self.cnn_weight * cnn_out + 
            self.vit_weight * vit_out
        )
        
        # Alternative: Concatenate and fuse
        # concat_features = torch.cat([cnn_out, vit_out], dim=1)
        # fusion_out = self.fusion(concat_features)
        
        return ensemble_logits, cnn_out, vit_out
```

### Training Configuration for RTX 4070
```python
# Optimal settings for 12GB VRAM
TRAINING_CONFIG = {
    'batch_size': 12,           # Optimized for 12GB VRAM
    'num_frames': 16,           # Frames per video
    'image_size': 224,          # Input resolution
    'mixed_precision': True,    # AMP for memory efficiency
    'gradient_accumulation': 4, # Effective batch size = 48
    'max_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler': 'cosine',
    'warmup_epochs': 10
}

# Data augmentation
AUGMENTATION_CONFIG = {
    'temporal_jitter': 0.1,
    'rotation_range': 15,
    'brightness_range': 0.2,
    'contrast_range': 0.2,
    'horizontal_flip': 0.5,
    'cutmix_prob': 0.3,
    'mixup_alpha': 0.2
}
```

---

## 💻 Installation & Setup

### Day 1 Installation Script
```bash
#!/bin/bash
# setup_google_asl.sh - Complete environment setup

echo "🚀 Setting up Google ASL Signs Recognition Project"

# Create conda environment
conda create -n google_asl python=3.9 -y
conda activate google_asl

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install additional ML libraries
pip install transformers timm
pip install opencv-python albumentations
pip install scikit-learn matplotlib seaborn
pip install gradio streamlit
pip install tensorboard wandb  # For experiment tracking
pip install tensorflow-datasets  # For Google ASL dataset

# Install video processing
pip install decord av  # Efficient video loading
pip install ffmpeg-python

# Install utilities
pip install tqdm rich  # Progress bars and pretty printing
pip install hydra-core  # Configuration management

# Create project structure
mkdir -p google_asl_recognition/{data,models,src,notebooks,demo,configs}
cd google_asl_recognition

echo "✅ Environment setup complete!"
echo "💡 Next: Download Google ASL Signs dataset"
```

### Dataset Download
```python
# download_dataset.py
import tensorflow_datasets as tfds
import os
from pathlib import Path

def download_google_asl():
    """Download and prepare Google ASL Signs dataset"""
    
    # Download dataset using TensorFlow Datasets
    print("📥 Downloading Google ASL Signs dataset...")
    
    # Load dataset
    ds_train = tfds.load(
        'asl_signs',
        split='train',
        data_dir='./data',
        download=True,
        as_supervised=False
    )
    
    ds_val = tfds.load(
        'asl_signs',
        split='validation',
        data_dir='./data',
        as_supervised=False
    )
    
    ds_test = tfds.load(
        'asl_signs',
        split='test',
        data_dir='./data',
        as_supervised=False
    )
    
    print("✅ Dataset downloaded successfully!")
    
    # Get dataset info
    info = tfds.builder('asl_signs').info
    print(f"📊 Dataset info:")
    print(f"   Total examples: {info.splits['train'].num_examples}")
    print(f"   Number of classes: {info.features['label'].num_classes}")
    print(f"   Video shape: {info.features['video'].shape}")
    
    return ds_train, ds_val, ds_test

if __name__ == "__main__":
    download_google_asl()
```

---

## 🎯 Real-time Demo Implementation

### Gradio Web Interface
```python
# demo/gradio_app.py
import gradio as gr
import torch
import cv2
import numpy as np
from collections import deque
import threading
import time

class GoogleASLDemo:
    def __init__(self, model_path='models/best_ensemble.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.class_names = self.load_class_names()
        self.frame_buffer = deque(maxlen=16)
        
    def load_model(self, path):
        """Load trained ensemble model"""
        checkpoint = torch.load(path, map_location=self.device)
        model = ASLEnsemble(
            cnn_model=ASLCNNModel(250),
            vit_model=VideoViT(250),
            num_classes=250
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def load_class_names(self):
        """Load ASL word vocabulary"""
        # Google ASL Signs vocabulary
        with open('data/class_names.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def preprocess_video(self, frames):
        """Preprocess video frames for inference"""
        processed_frames = []
        
        for frame in frames:
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224))
            # Normalize
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            processed_frames.append(frame)
        
        # Convert to tensor: (1, frames, channels, height, width)
        video_tensor = torch.tensor(processed_frames).permute(3, 0, 1, 2).unsqueeze(0)
        return video_tensor.to(self.device)
    
    def predict(self, video_frames):
        """Make prediction on video frames"""
        if len(video_frames) < 16:
            return None, 0.0, []
        
        # Take last 16 frames
        frames = list(video_frames)[-16:]
        
        # Preprocess
        video_tensor = self.preprocess_video(frames)
        
        # Inference
        with torch.no_grad():
            ensemble_out, cnn_out, vit_out = self.model(video_tensor)
            probabilities = torch.softmax(ensemble_out, dim=1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                predictions.append({
                    'word': self.class_names[idx.item()],
                    'confidence': prob.item(),
                    'cnn_conf': torch.softmax(cnn_out, dim=1)[0, idx].item(),
                    'vit_conf': torch.softmax(vit_out, dim=1)[0, idx].item()
                })
        
        top_prediction = predictions[0]
        return top_prediction['word'], top_prediction['confidence'], predictions
    
    def process_webcam_frame(self, frame):
        """Process single webcam frame"""
        if frame is None:
            return frame, "No prediction", ""
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Make prediction if buffer is full
        if len(self.frame_buffer) == 16:
            word, confidence, top_predictions = self.predict(self.frame_buffer)
            
            if word:
                # Draw prediction on frame
                cv2.putText(
                    frame, 
                    f"{word} ({confidence:.2%})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Create top predictions text
                pred_text = "Top 5 Predictions:\n"
                for i, pred in enumerate(top_predictions[:5]):
                    pred_text += f"{i+1}. {pred['word']}: {pred['confidence']:.1%}\n"
                
                return frame, f"{word}", pred_text
        
        return frame, "Processing...", "Collecting frames..."
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="Google ASL Signs Recognition") as interface:
            gr.Markdown("# 🤟 Google ASL Signs Recognition")
            gr.Markdown("**Real-time ASL word recognition using CNN + Vision Transformer ensemble**")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Webcam input
                    webcam = gr.Image(
                        source="webcam",
                        streaming=True,
                        label="Webcam Feed"
                    )
                
                with gr.Column(scale=1):
                    # Current prediction
                    current_pred = gr.Textbox(
                        label="Current Prediction",
                        value="Ready...",
                        interactive=False
                    )
                    
                    # Top predictions
                    top_preds = gr.Textbox(
                        label="Top 5 Predictions",
                        value="Start signing to see predictions",
                        lines=6,
                        interactive=False
                    )
            
            # Model info
            with gr.Row():
                gr.Markdown("""
                ### 📊 Model Information
                - **Dataset:** Google ASL Signs (250 words)
                - **Architecture:** CNN + Vision Transformer Ensemble
                - **Accuracy:** 80-85% validation
                - **Inference Time:** ~30-50ms on RTX 4070
                """)
            
            # Categories showcase
            with gr.Row():
                gr.Markdown("""
                ### 📚 Recognized Words (250 total)
                **Sample categories:** Numbers, Letters, Common Words, Phrases, Actions, Objects, Colors, Family, Food, etc.
                
                **Popular words:** HELLO, THANK_YOU, PLEASE, YES, NO, WATER, FOOD, LOVE, HAPPY, etc.
                """)
            
            # Process webcam stream
            webcam.stream(
                self.process_webcam_frame,
                inputs=[webcam],
                outputs=[webcam, current_pred, top_preds],
                show_progress=False
            )
        
        return interface

def main():
    demo = GoogleASLDemo()
    interface = demo.create_interface()
    
    # Launch interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Create public link
        debug=True
    )

if __name__ == "__main__":
    main()
```

---

## 📊 Performance Optimization

### RTX 4070 Specific Optimizations
```python
# src/train_optimized.py
import torch
import torch.amp as amp
from torch.utils.data import DataLoader
import time

class OptimizedTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.scaler = amp.GradScaler()  # Mixed precision
        
        # Compile model for PyTorch 2.0+ (significant speedup)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Optimize CUDA settings for RTX 4070
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Optimized training loop"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos, labels = videos.to(self.device), labels.to(self.device)
            
            # Mixed precision forward pass
            with amp.autocast():
                outputs = self.model(videos)
                if isinstance(outputs, tuple):  # Ensemble model
                    outputs = outputs[0]  # Main prediction
                loss = criterion(outputs, labels)
            
            # Mixed precision backward pass
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Memory management every 50 batches
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy

# Temperature monitoring
def monitor_gpu_temp():
    """Monitor GPU temperature during training"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            'temperature': temp,
            'memory_used': memory_info.used // 1024**2,  # MB
            'memory_total': memory_info.total // 1024**2,  # MB
            'memory_percent': memory_info.used / memory_info.total * 100
        }
    except:
        return None

# Automatic batch size finder
def find_optimal_batch_size(model, input_shape=(3, 16, 224, 224)):
    """Find optimal batch size for RTX 4070"""
    model.train()
    
    batch_size = 1
    max_batch_size = 1
    
    while batch_size <= 64:  # Test up to batch size 64
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Test batch
            dummy_input = torch.randn(batch_size, *input_shape).cuda()
            dummy_target = torch.randint(0, 250, (batch_size,)).cuda()
            
            # Forward pass
            output = model(dummy_input)
            if isinstance(output, tuple):
                output = output[0]
            
            loss = torch.nn.CrossEntropyLoss()(output, dummy_target)
            loss.backward()
            
            max_batch_size = batch_size
            print(f"✅ Batch size {batch_size} works")
            batch_size *= 2
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ Batch size {batch_size} failed")
            break
        except Exception as e:
            print(f"❌ Error at batch size {batch_size}: {e}")
            break
    
    optimal_batch_size = max_batch_size // 2  # Use 50% of max for safety
    print(f"🎯 Optimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size
```