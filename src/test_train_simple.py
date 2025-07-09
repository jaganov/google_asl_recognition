#!/usr/bin/env python3
"""
Упрощенный тест train.py без долгих операций
"""

import torch
import sys
import os

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import ASLTrainer
from data_loader import ASLDataLoader
from preprocessing import ASLPreprocessor
from augmentations import ASLAugmentations

def test_train_simple():
    """Упрощенный тест train.py"""
    print("🧪 Упрощенный тест train.py...")
    
    try:
        # 1. Тест аугментаций
        print("\n1. Тестируем аугментации...")
        augmenter = ASLAugmentations()
        test_features = torch.randn(2, 10, 468 * 3 * 3)  # Маленький размер для быстрого теста
        augmented = augmenter(test_features)
        print(f"   ✅ Аугментации работают: {test_features.shape} -> {augmented.shape}")
        
        # 2. Тест препроцессора
        print("\n2. Тестируем препроцессор...")
        preprocessor = ASLPreprocessor(max_len=32)  # Маленькая длина
        test_landmarks = torch.randn(2, 20, 468, 3)
        features = preprocessor(test_landmarks)
        print(f"   ✅ Препроцессор работает: {test_landmarks.shape} -> {features.shape}")
        
        # 3. Тест тренера с минимальными настройками
        print("\n3. Тестируем тренер с минимальными настройками...")
        trainer = ASLTrainer(
            data_dir="../data/google_asl_signs",
            model_dir="models/test",
            max_len=32,  # Маленькая длина
            batch_size=2,  # Маленький batch
            dim=32,  # Маленькая модель
            lr=1e-3,
            epochs=1,  # Только 1 эпоха
            use_augmentations=True,
            use_mixed_precision=True,
            gradient_clip_val=1.0,
            gradient_accumulation_steps=1  # Без accumulation для простоты
        )
        
        print(f"   ✅ Тренер создан успешно")
        print(f"   ✅ Модель параметров: {sum(p.numel() for p in trainer.model.parameters()):,}")
        print(f"   ✅ Mixed precision: {'Включен' if trainer.use_mixed_precision else 'Отключен'}")
        
        # 4. Тест только одного батча (без полной эпохи)
        print("\n4. Тестируем один батч...")
        try:
            # Берем только один батч
            sample_batch = next(iter(trainer.train_loader))
            features = sample_batch['features'].to(trainer.device, non_blocking=True)
            labels = sample_batch['labels'].to(trainer.device, non_blocking=True)
            
            # Forward pass
            if trainer.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = trainer.model(features)
                    loss = trainer.criterion(outputs, labels)
            else:
                outputs = trainer.model(features)
                loss = trainer.criterion(outputs, labels)
            
            print(f"   ✅ Forward pass работает: Loss={loss.item():.4f}")
            print(f"   ✅ Output shape: {outputs.shape}")
            
        except Exception as e:
            print(f"   ⚠️ Forward pass не работает: {e}")
            return False
        
        # 5. Тест информации о GPU
        if torch.cuda.is_available():
            print("\n5. Информация о GPU:")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ GPU: {gpu_name}")
            print(f"   ✅ Память: {gpu_memory:.1f} GB")
        
        print("\n🎉 Упрощенный тест прошел успешно!")
        print("✅ train.py готов к использованию")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка в тесте: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_train_simple()
    if success:
        print("\n🚀 Можно запускать обучение!")
        print("python train.py")
    else:
        print("\n⚠️ Нужно исправить ошибки перед запуском обучения") 