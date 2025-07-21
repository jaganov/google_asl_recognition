# 🔧 Установка и настройка

Подробное руководство по установке и настройке проекта Google ASL Recognition.

## 🎯 Системные требования

### Минимальные требования
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.8+
- **GPU**: NVIDIA с CUDA поддержкой (4GB+ VRAM)
- **RAM**: 8GB
- **Storage**: 10GB свободного места

### Рекомендуемые требования
- **OS**: Windows 11, Ubuntu 20.04+
- **Python**: 3.10+
- **GPU**: RTX4070 или лучше (12GB+ VRAM)
- **RAM**: 16GB+
- **Storage**: 20GB свободного места
- **Webcam**: HD камера для живого распознавания

## 🚀 Пошаговая установка

### Шаг 1: Подготовка системы

#### Windows
```bash
# Установка Git (если не установлен)
# Скачайте с https://git-scm.com/download/win

# Установка Python (если не установлен)
# Скачайте с https://www.python.org/downloads/
# Убедитесь, что отмечен "Add Python to PATH"
```

#### Linux (Ubuntu/Debian)
```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка зависимостей
sudo apt install -y python3 python3-pip python3-venv git

# Установка CUDA (если не установлен)
# Следуйте инструкциям: https://developer.nvidia.com/cuda-downloads
```

#### macOS
```bash
# Установка Homebrew (если не установлен)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установка Python
brew install python@3.10

# Установка Git
brew install git
```

### Шаг 2: Клонирование репозитория

```bash
# Клонирование
git clone <repository-url>
cd google_asl_recognition

# Проверка структуры
ls -la
```

### Шаг 3: Создание виртуального окружения

#### Windows
```bash
# Создание виртуального окружения
python -m venv .venv

# Активация
.venv\Scripts\activate

# Проверка
python --version
pip --version
```

#### Linux/macOS
```bash
# Создание виртуального окружения
python3 -m venv .venv

# Активация
source .venv/bin/activate

# Проверка
python --version
pip --version
```

### Шаг 4: Установка PyTorch

#### С CUDA поддержкой (рекомендуется)
```bash
# PyTorch с CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Проверка CUDA
python -c "import torch; print(f'CUDA доступен: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

#### CPU только (не рекомендуется)
```bash
# PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Шаг 5: Установка зависимостей

```bash
# Обновление pip
pip install --upgrade pip

# Установка зависимостей
pip install -r requirements.txt
```

### Шаг 6: Проверка установки

```bash
# Тест импорта основных библиотек
python -c "
import torch
import cv2
import mediapipe
import numpy as np
import pandas as pd
print('✅ Все библиотеки установлены успешно!')
"

# Проверка GPU
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️ CUDA недоступен, будет использоваться CPU')
"
```

## 🎮 Настройка для живого распознавания

### Шаг 1: Тест камеры

```bash
cd manual
python test_camera.py
```

**Ожидаемый результат:**
- Открывается окно с камерой
- Видно изображение с камеры
- Нажатие 'q' закрывает окно

### Шаг 2: Запуск живого распознавания

```bash
# Запуск с предобученной моделью
python step5_live_recognition.py
```

**Ожидаемый результат:**
- Открывается окно с камерой
- Отображаются landmarks MediaPipe
- Показываются предсказания жестов

## 📊 Установка датасета (опционально)

### Для тренировки собственной модели

```bash
# Скачивание Google ASL Signs dataset
# Следуйте инструкциям в docs/data-preparation.md

# Подготовка данных
python step1_extract_words.py
python step1.2_split_train_test.py
python step2_prepare_dataset.py
```

## 🔧 Дополнительные настройки

### Настройка CUDA

#### Проверка версии CUDA
```bash
nvidia-smi
nvcc --version
```

#### Переменные окружения
```bash
# Добавьте в ~/.bashrc или ~/.zshrc
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
```

### Настройка производительности

#### Windows
```bash
# Включение режима производительности
# Панель управления NVIDIA -> Управление настройками 3D -> Режим управления питанием -> Предпочтение максимальной производительности
```

#### Linux
```bash
# Установка nvidia-settings
sudo apt install nvidia-settings

# Настройка производительности
nvidia-settings
```

## 🐛 Устранение проблем

### Проблема: CUDA недоступен

#### Решение 1: Проверка драйверов
```bash
# Windows
nvidia-smi

# Linux
nvidia-smi
nvcc --version
```

#### Решение 2: Переустановка PyTorch
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Проблема: Камера не работает

#### Решение 1: Проверка разрешений
```bash
# Linux
sudo usermod -a -G video $USER

# macOS
# Системные настройки -> Безопасность и конфиденциальность -> Камера
```

#### Решение 2: Другой ID камеры
```bash
python step5_live_recognition.py --camera_id 1
```

### Проблема: Out of Memory

#### Решение 1: Уменьшение batch size
```bash
# В коде измените batch_size на меньшее значение
batch_size = 16  # вместо 32
```

#### Решение 2: Очистка памяти
```python
import torch
torch.cuda.empty_cache()
```

### Проблема: Медленная производительность

#### Решение 1: Проверка оптимизаций
```python
# Убедитесь, что включены оптимизации
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

#### Решение 2: Использование mixed precision
```python
from torch.cuda.amp import autocast, GradScaler
# См. docs/rtx4070-optimizations.md
```

## 📚 Проверка установки

### Полная проверка
```bash
# Создайте файл test_installation.py
python test_installation.py
```

**Содержимое test_installation.py:**
```python
import sys
import torch
import cv2
import mediapipe
import numpy as np
import pandas as pd

def test_installation():
    print("🔍 Проверка установки Google ASL Recognition")
    print("=" * 50)
    
    # Python версия
    print(f"✅ Python: {sys.version}")
    
    # PyTorch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA доступен: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # OpenCV
    print(f"✅ OpenCV: {cv2.__version__}")
    
    # MediaPipe
    print(f"✅ MediaPipe: {mediapipe.__version__}")
    
    # NumPy
    print(f"✅ NumPy: {np.__version__}")
    
    # Pandas
    print(f"✅ Pandas: {pd.__version__}")
    
    print("=" * 50)
    print("🎉 Установка завершена успешно!")
    print("🚀 Готов к использованию!")

if __name__ == "__main__":
    test_installation()
```

## 🎯 Следующие шаги

После успешной установки:

1. **Быстрый старт**: [docs/quickstart.md](quickstart.md)
2. **Живое распознавание**: [docs/live-recognition.md](live-recognition.md)
3. **Тренировка модели**: [docs/training.md](training.md)
4. **Оптимизации**: [docs/rtx4070-optimizations.md](rtx4070-optimizations.md)

## 📞 Поддержка

Если у вас возникли проблемы:

1. Проверьте раздел "Устранение проблем" выше
2. Изучите документацию в папке `docs/`
3. Создайте issue в репозитории с подробным описанием проблемы

---

**Установка завершена! Готов к распознаванию жестов! 🤟** 