# 📋 Система манифестов для ASL моделей

Система манифестов автоматически создает подробные описания каждой обученной модели с префиксом по дате для отслеживания истории тренировок и экспериментов.

## 🎯 Обзор

Система манифестов обеспечивает:
- **Отслеживание истории** - каждая модель имеет уникальный идентификатор
- **Детальную документацию** - все параметры и результаты сохраняются
- **Сравнение версий** - легко сравнивать разные подходы
- **Воспроизводимость** - полная информация для воспроизведения результатов
- **Анализ трендов** - возможность анализа улучшений

## 📁 Структура файлов

После каждой тренировки создаются следующие файлы в папке `models/`:

```
models/
├── asl_model_v20250720_080209.pth              # Модель
├── asl_model_v20250720_080209_manifest.json    # Манифест лучшей модели
└── asl_model_v20250720_080209_final_manifest.json  # Финальный манифест
```

## 🏷️ Формат именования

- **Префикс**: `asl_model_v`
- **Дата**: `YYYYMMDD_HHMMSS`
- **Пример**: `asl_model_v20250720_080209`

## 📄 Типы манифестов

### 1. Манифест лучшей модели (`*_manifest.json`)

Создается каждый раз, когда находится лучшая модель во время тренировки.

**Содержит:**
- Информацию о модели и архитектуре
- Параметры тренировки
- Текущие результаты (на момент сохранения)

### 2. Финальный манифест (`*_final_manifest.json`)

Создается в конце тренировки с полными результатами.

**Дополнительно содержит:**
- Время тренировки
- Полную историю тренировки
- Информацию о ранней остановке

## 🏗️ Структура манифеста

```json
{
  "model_info": {
    "name": "asl_model_v20250720_080209",
    "timestamp": "20250720_080209",
    "architecture": "Enhanced_TCN_LSTM_Transformer_v2",
    "version": "adaptive_regularization_v2",
    "description": "ASL Recognition model with adaptive regularization"
  },
  "model_parameters": {
    "total_params": 1968601,
    "trainable_params": 1968601,
    "input_dim": 744,
    "hidden_dim": 192,
    "num_classes": 25,
    "max_sequence_length": 384
  },
  "training_config": {
    "epochs": 300,
    "batch_size": 32,
    "learning_rate": 0.0004,
    "optimizer": "AdamW",
    "weight_decay": 0.005,
    "loss_function": "CrossEntropyLoss",
    "label_smoothing": 0.05,
    "scheduler": "CosineAnnealingWarmRestarts"
  },
  "architecture_details": {
    "preprocessing": {
      "type": "PreprocessingLayer",
      "max_len": 384,
      "motion_features": [
        "velocity",
        "acceleration", 
        "relative_motion",
        "temporal_consistency",
        "motion_magnitude",
        "motion_direction"
      ]
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
    }
  },
  "training_results": {
    "best_epoch": 150,
    "best_val_accuracy": 75.76,
    "best_train_accuracy": 85.2,
    "current_train_loss": 0.8,
    "current_val_loss": 1.2
  },
  "final_results": {
    "training_time_hours": 1.75,
    "best_val_accuracy": 75.76,
    "improvement_over_baseline": 10.76,
    "total_epochs_trained": 150,
    "early_stopping_triggered": true,
    "training_history": {
      "train_losses": [...],
      "train_accuracies": [...],
      "val_losses": [...],
      "val_accuracies": [...]
    }
  }
}
```

## 🚀 Использование

### Запуск тренировки
```bash
cd manual
python step3_prepare_train.py
```

### Просмотр результатов
```bash
# Посмотреть все модели
ls models/asl_model_v*.pth

# Посмотреть манифест конкретной модели
cat models/asl_model_v20250720_080209_manifest.json | jq

# Сравнить результаты разных версий
cat models/*_final_manifest.json | jq '.final_results.best_val_accuracy'
```

### Анализ истории
```bash
# Найти лучшую модель
cat models/*_final_manifest.json | jq -r '.model_info.name + ": " + (.final_results.best_val_accuracy | tostring) + "%"' | sort -k2 -nr

# Время тренировки
cat models/*_final_manifest.json | jq -r '.model_info.name + ": " + (.final_results.training_time_hours | tostring) + "h"'
```

## 💻 Программное использование

### Загрузка модели с манифестом
```python
import torch
import json

# Загружаем модель
checkpoint = torch.load('models/asl_model_v20250720_080209.pth')
model_state = checkpoint['model_state_dict']
manifest = checkpoint['manifest']

# Восстанавливаем архитектуру
model = ASLModel(
    input_dim=manifest['model_parameters']['input_dim'],
    num_classes=manifest['model_parameters']['num_classes'],
    dim=manifest['model_parameters']['hidden_dim']
)
model.load_state_dict(model_state)

print(f"Модель {manifest['model_info']['name']}")
print(f"Точность: {manifest['training_results']['best_val_accuracy']:.2f}%")
```

### Анализ прогресса
```python
import glob
import json

# Анализируем все модели
manifests = []
for manifest_file in glob.glob('models/*_final_manifest.json'):
    with open(manifest_file, 'r') as f:
        manifests.append(json.load(f))

# Сортируем по точности
manifests.sort(key=lambda x: x['final_results']['best_val_accuracy'], reverse=True)

print("Топ-5 моделей:")
for i, manifest in enumerate(manifests[:5]):
    print(f"{i+1}. {manifest['model_info']['name']}: {manifest['final_results']['best_val_accuracy']:.2f}%")
```

### Сравнение архитектур
```python
def compare_architectures():
    """Сравнивает разные архитектуры моделей"""
    architectures = {}
    
    for manifest_file in glob.glob('models/*_final_manifest.json'):
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            
        arch = manifest['model_info']['architecture']
        accuracy = manifest['final_results']['best_val_accuracy']
        
        if arch not in architectures:
            architectures[arch] = []
        architectures[arch].append(accuracy)
    
    print("Сравнение архитектур:")
    for arch, accuracies in architectures.items():
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"{arch}: {avg_acc:.2f}% (среднее)")
```

## 📊 Анализ результатов

### Ключевые метрики
- **best_val_accuracy** - лучшая точность на валидации
- **training_time_hours** - время тренировки
- **total_epochs_trained** - количество эпох
- **early_stopping_triggered** - сработала ли ранняя остановка

### Тренды улучшений
```python
def analyze_improvements():
    """Анализирует улучшения между версиями"""
    manifests = []
    for manifest_file in glob.glob('models/*_final_manifest.json'):
        with open(manifest_file, 'r') as f:
            manifests.append(json.load(f))
    
    # Сортируем по времени создания
    manifests.sort(key=lambda x: x['model_info']['timestamp'])
    
    print("История улучшений:")
    for i in range(1, len(manifests)):
        prev_acc = manifests[i-1]['final_results']['best_val_accuracy']
        curr_acc = manifests[i]['final_results']['best_val_accuracy']
        improvement = curr_acc - prev_acc
        
        print(f"{manifests[i]['model_info']['name']}: +{improvement:.2f}%")
```

## 🔧 Настройка

### Создание кастомного манифеста
```python
def create_custom_manifest(model, results, config):
    """Создает кастомный манифест"""
    manifest = {
        "model_info": {
            "name": f"custom_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "architecture": "Custom_Architecture",
            "version": "custom_v1",
            "description": "Custom model with specific configuration"
        },
        "model_parameters": {
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input_dim": config['input_dim'],
            "hidden_dim": config['hidden_dim'],
            "num_classes": config['num_classes']
        },
        "training_config": config,
        "results": results
    }
    
    return manifest
```

## 📈 Визуализация

### График прогресса
```python
import matplotlib.pyplot as plt

def plot_training_progress(manifest_file):
    """Строит график прогресса тренировки"""
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    history = manifest['final_results']['training_history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Accuracy
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

## 🎯 Преимущества

1. **Полная документация** - все детали эксперимента сохраняются
2. **Воспроизводимость** - можно точно воспроизвести любой эксперимент
3. **Сравнение** - легко сравнивать разные подходы
4. **Анализ** - возможность анализа трендов и улучшений
5. **Отслеживание** - история всех экспериментов

## 📚 Дополнительные ресурсы

- [JSON Schema](https://json-schema.org/) - для валидации манифестов
- [jq](https://stedolan.github.io/jq/) - для работы с JSON в командной строке
- [matplotlib](https://matplotlib.org/) - для визуализации результатов

---

**Система манифестов обеспечивает полную прозрачность и воспроизводимость экспериментов! 📋** 