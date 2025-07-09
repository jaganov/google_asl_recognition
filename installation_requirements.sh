#!/bin/bash
# install_asl_uv.sh - UV installation for Python 3.12

echo "🚀 Installing Google ASL Dependencies with UV..."

# Make sure you're in your venv
echo "Current Python: $(python --version)"
echo "Current pip location: $(which pip)"

# Step 1: Install PyTorch with CUDA 12.1 (Python 3.12 compatible)
echo "📦 Installing PyTorch with CUDA 12.1..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install Vision & ML libraries
echo "🤖 Installing ML frameworks..."
uv pip install transformers timm accelerate

# Step 3: Install Computer Vision
echo "👁️ Installing CV libraries..."
uv pip install opencv-python albumentations kornia

# Step 4: Install Video Processing (critical for ASL)
echo "🎥 Installing video processing..."
uv pip install decord av ffmpeg-python

# Step 5: Install Data Science stack
echo "📊 Installing data tools..."
uv pip install numpy pandas scikit-learn matplotlib seaborn plotly

# Step 6: Install Dataset tools
echo "📚 Installing dataset tools..."
uv pip install datasets tensorflow-datasets h5py

# Step 7: Install Web Interface
echo "🌐 Installing demo tools..."
uv pip install gradio streamlit fastapi uvicorn

# Step 8: Install utilities
echo "🛠️ Installing utilities..."
uv pip install tqdm rich wandb tensorboard hydra-core pynvml psutil

echo "✅ UV Installation complete!"
echo "🎯 Ready for Google ASL training with Python 3.12!"