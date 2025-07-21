# 📚 Google ASL Recognition - Documentation

Welcome to the comprehensive documentation for the Google ASL Recognition project! This documentation covers everything you need to know about training, deploying, and using the ASL recognition system.

## 🎯 Project Overview

This project implements a state-of-the-art ASL recognition model based on the winning solution from Google's ASL Signs competition. The system recognizes **25 common ASL gestures** with **76.05% accuracy** and provides real-time recognition through webcam input.

### 🏆 Key Features
- **25 ASL Gestures**: hello, please, thankyou, bye, mom, dad, boy, girl, man, child, drink, sleep, go, happy, sad, hungry, thirsty, sick, bad, red, blue, green, yellow, black, white
- **Real-time Recognition**: Live gesture recognition via webcam
- **RTX4070 Optimized**: Special optimizations for modern GPUs
- **Comprehensive Training**: Full training pipeline with manifest system
- **High Accuracy**: 76.05% validation accuracy on the test set

## 📖 Documentation Structure

### 🚀 Getting Started
- **[Quick Start Guide](quickstart.md)** - Get up and running in minutes
- **[Installation Guide](installation.md)** - Complete setup instructions
- **[Data Preparation](data-preparation.md)** - Prepare the Google ASL Signs dataset

### 🔧 Core Components
- **[Model Architecture](architecture.md)** - Detailed model design and components
- **[Training Guide](training.md)** - How to train the ASL recognition model
- **[Live Recognition](live-recognition.md)** - Real-time gesture recognition setup
- **[RTX4070 Optimizations](rtx4070-optimizations.md)** - GPU-specific performance optimizations

### 📊 Results & Analysis
- **[Model Results](model-results.md)** - Performance analysis and benchmarks
- **[Manifest System](manifest-system.md)** - Model versioning and experiment tracking

### 🛠️ Advanced Topics
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[API Reference](api-reference.md)** - Function documentation
- **[Examples](examples.md)** - Practical usage examples

## 🎮 Quick Start

### 1. Installation
```bash
git clone <repository-url>
cd google_asl_recognition
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
cd manual
python step1_extract_words.py
python step1.2_split_train_test.py
python step2_prepare_dataset.py
```

### 3. Training
```bash
python step3_prepare_train.py
```

### 4. Live Recognition
```bash
python step5_live_recognition.py
```

## 📈 Latest Model Performance

**Model**: `asl_model_v20250720_080209`
- **Architecture**: Enhanced TCN + LSTM + Transformer v2
- **Validation Accuracy**: 76.05%
- **Parameters**: 1.97M
- **Training Time**: ~1.75 hours
- **Classes**: 25 ASL gestures

## 🏗️ Model Architecture

The model uses a hybrid architecture combining:
- **TCN Blocks** (3 layers with dilations 1,2,4) - Local temporal patterns
- **Bidirectional LSTM** (2 layers) - Long-term dependencies
- **Temporal Attention** (8 heads) - Focus on important frames
- **Conv1D + Transformer** - Final processing and classification
- **Adaptive Regularization** - Overfitting prevention

## 🎯 Recognized Gestures

The system recognizes 25 common ASL gestures:

**Greetings & Courtesy**: hello, please, thankyou, bye  
**Family**: mom, dad, boy, girl, man, child  
**Actions**: drink, sleep, go  
**Emotions**: happy, sad, hungry, thirsty, sick, bad  
**Colors**: red, blue, green, yellow, black, white  

## 🔧 Main Scripts

### Data Preparation
- **`step1_extract_words.py`** - Extract 25 signs from the full dataset
- **`step1.2_split_train_test.py`** - Create train/test splits by participant
- **`step2_prepare_dataset.py`** - Prepare PyTorch tensors with preprocessing

### Training
- **`step3_prepare_train.py`** - Main training script with advanced architecture
- **`rtx4070_optimizations.py`** - GPU-specific optimizations

### Live Recognition
- **`step5_live_recognition.py`** - Real-time ASL recognition via webcam
- **`test_camera.py`** - Camera testing utility

## 📊 Model Files

The trained model and metadata are stored in `manual/models/`:
- **`asl_model_v20250720_080209.pth`** - Model weights (23MB)
- **`asl_model_v20250720_080209_manifest.json`** - Training manifest
- **`asl_model_v20250720_080209_final_manifest.json`** - Final results and analysis

## 🎥 Live Recognition Features

### MediaPipe Integration
- **543 landmarks per frame**: Face (468) + Pose (33) + Hands (42)
- **Real-time processing**: <50ms per prediction
- **Visual feedback**: Live landmark visualization

### Controls
- **'q'**: Stop recognition
- **'s'**: Save screenshot
- **'r'**: Reset frame buffer
- **'h'**: Show/hide help

## 🤝 Contributing

We welcome contributions! Please see the [Contributing Guide](../CONTRIBUTING.md) for details.

## 📄 License

This project is based on the winning solution from Google's ASL Signs competition. Please refer to the original competition terms and conditions.

## 🙏 Acknowledgments

- **Google ASL Signs Competition** - Dataset and inspiration
- **MediaPipe** - Hand tracking and landmark extraction
- **PyTorch Community** - Deep learning framework
- **Competition Winners** - Model architecture inspiration

---

**Ready to start recognizing ASL gestures! 🤟**

For detailed information on any topic, please refer to the specific documentation files listed above. 