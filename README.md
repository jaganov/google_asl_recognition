
## Install
```bash
# Убедитесь что venv активирован
source .venv/bin/activate  # или .venv\Scripts\activate на Windows

# Установка PyTorch с CUDA 12.1 (совместимо с Python 3.12)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Установка всех остальных пакетов одной командой
uv pip install transformers timm opencv-python albumentations decord av gradio wandb tensorboard datasets tensorflow-datasets rich tqdm hydra-core pynvml accelerate kornia matplotlib seaborn plotly
```