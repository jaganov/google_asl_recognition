# 🚀 Быстрый старт

Краткое руководство по запуску проекта Google ASL Recognition за 10 минут.

## ⚡ Установка (5 минут)

### 1. Клонирование и настройка
```bash
git clone <repository-url>
cd google_asl_recognition

# Создание виртуального окружения
python -m venv .venv

# Активация (Windows)
.venv\Scripts\activate

# Активация (Linux/Mac)
source .venv/bin/activate
```

### 2. Установка зависимостей
```bash
# PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Остальные зависимости
pip install -r requirements.txt
```

## 🎯 Быстрый тест (3 минуты)

### 1. Проверка GPU
```python
import torch
print(f"CUDA доступен: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 2. Тест камеры
```bash
cd manual
python test_camera.py
```

## 🎮 Запуск живого распознавания (2 минуты)

### Использование предобученной модели
```bash
python step5_live_recognition.py
```

**Управление:**
- **'q'** - выход
- **'s'** - сохранить скриншот
- **'r'** - сбросить буфер
- **'h'** - показать помощь

## 📊 Распознаваемые жесты

**25 жестов ASL:**
- **Приветствия**: hello, please, thankyou, bye
- **Семья**: mom, dad, boy, girl, man, child
- **Действия**: drink, sleep, go
- **Эмоции**: happy, sad, hungry, thirsty, sick, bad
- **Цвета**: red, blue, green, yellow, black, white

## 🎯 Ожидаемые результаты

- **Точность**: 75.76% на валидационном наборе
- **FPS**: 15-30 кадров/сек в реальном времени
- **Задержка**: 100-200ms на предсказание
- **Память**: ~2-4GB GPU

## 🔧 Требования

### Минимальные
- **GPU**: NVIDIA с CUDA поддержкой
- **RAM**: 8GB
- **Python**: 3.8+

### Рекомендуемые
- **GPU**: RTX4070 или лучше
- **RAM**: 16GB+
- **Python**: 3.10+

## 🐛 Быстрое устранение проблем

### Камера не работает
```bash
# Попробуйте другой ID камеры
python step5_live_recognition.py --camera_id 1
```

### Out of Memory
```bash
# Уменьшите batch size
python step5_live_recognition.py --target_frames 8
```

### Низкая точность
- Улучшите освещение
- Убедитесь, что лицо и руки видны
- Уменьшите расстояние до камеры

## 📚 Следующие шаги

1. **Изучите документацию**: [docs/README.md](README.md)
2. **Настройте тренировку**: [docs/training.md](training.md)
3. **Подготовьте данные**: [docs/data-preparation.md](data-preparation.md)
4. **Оптимизируйте производительность**: [docs/rtx4070-optimizations.md](rtx4070-optimizations.md)

## 🎉 Готово!

Вы успешно запустили систему распознавания ASL жестов! 

**Попробуйте показать жесты:**
- 👋 hello (приветствие)
- 🙏 please (пожалуйста)
- 👍 thankyou (спасибо)
- 👋 bye (до свидания)

---

**Удачи в изучении ASL! 🤟** 