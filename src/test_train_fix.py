#!/usr/bin/env python3
"""
Тест исправленного train.py с оптимизациями RTX 4070 (Windows-совместимый)
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

# Сброс флага для чистого теста
if hasattr(ASLPreprocessor, '_landmarks_printed'):
    delattr(ASLPreprocessor, '_landmarks_printed')

def test_train_fix():
    """Тест исправленного train.py с оптимизациями (Windows-совместимый)"""
    print("🧪 Тестируем исправленный train.py с оптимизациями RTX 4070 (Windows)...")
    
    try:
        # 1. Тест аугментаций
        print("\n1. Тестируем аугментации...")
        augmenter = ASLAugmentations()
        test_features = torch.randn(2, 50, 468 * 3 * 3)  # (batch, frames, features)
        augmented = augmenter(test_features)
        print(f"   ✅ Аугментации работают: {test_features.shape} -> {augmented.shape}")
        
        # 2. Тест препроцессора (убираем отдельный тест)
        print("\n2. Тест препроцессора будет выполнен в DataLoader...")
        
        # 3. Тест DataLoader с аугментациями
        print("\n3. Тестируем DataLoader с аугментациями...")
        dataloader = ASLDataLoader(
            data_dir="../data/google_asl_signs",
            batch_size=4,
            max_len=64,
            preprocessor=None  # DataLoader создаст препроцессор сам
        )
        
        # Получаем DataLoader'ы с аугментациями
        train_loader, val_loader, test_loader = dataloader.get_dataloaders(
            augment_train=True
        )
        
        # Тестируем один батч
        sample_batch = next(iter(train_loader))
        print(f"   ✅ DataLoader работает: {sample_batch['features'].shape}")
        print(f"   ✅ Ключи батча: {list(sample_batch.keys())}")
        
        # 4. Тест тренера с оптимизациями (Windows-совместимый)
        print("\n4. Тестируем тренер с оптимизациями RTX 4070 (Windows)...")
        trainer = ASLTrainer(
            data_dir="../data/google_asl_signs",
            model_dir="models/test",
            max_len=64,
            batch_size=4,
            dim=64,  # Маленькая модель для теста
            lr=1e-3,
            epochs=2,
            use_augmentations=True,
            use_mixed_precision=True,  # Тестируем mixed precision
            gradient_clip_val=1.0,     # Тестируем gradient clipping
            gradient_accumulation_steps=2  # Тестируем gradient accumulation
        )
        
        print(f"   ✅ Тренер создан успешно")
        print(f"   ✅ Модель параметров: {sum(p.numel() for p in trainer.model.parameters()):,}")
        print(f"   ✅ Mixed precision: {'Включен' if trainer.use_mixed_precision else 'Отключен'}")
        print(f"   ✅ Gradient clipping: {trainer.gradient_clip_val}")
        print(f"   ✅ Gradient accumulation: {trainer.gradient_accumulation_steps} steps")
        
        # 5. Тест одной эпохи обучения с оптимизациями
        print("\n5. Тестируем одну эпоху обучения с оптимизациями...")
        try:
            train_loss, train_acc = trainer.train_epoch(0)
            print(f"   ✅ Обучение работает: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        except Exception as e:
            print(f"   ⚠️ Ошибка в обучении: {e}")
            print("   ℹ️ Это может быть связано с Windows-специфичными проблемами")
            return False
        
        # 6. Тест валидации с оптимизациями
        print("\n6. Тестируем валидацию с оптимизациями...")
        try:
            val_loss, val_acc = trainer.validate(0)
            print(f"   ✅ Валидация работает: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        except Exception as e:
            print(f"   ⚠️ Ошибка в валидации: {e}")
            print("   ℹ️ Это может быть связано с Windows-специфичными проблемами")
            return False
        
        # 7. Тест информации о GPU
        if torch.cuda.is_available():
            print("\n7. Информация о GPU:")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ GPU: {gpu_name}")
            print(f"   ✅ Память: {gpu_memory:.1f} GB")
            
            # Тест памяти во время обучения
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"   ✅ Использовано: {gpu_memory_used:.1f} GB")
            print(f"   ✅ Зарезервировано: {gpu_memory_cached:.1f} GB")
        
        # 8. Тест PyTorch compile (если доступно)
        import platform
        if hasattr(torch, 'compile'):
            print("\n8. PyTorch 2.0+ compile:")
            print(f"   ✅ PyTorch compile доступен")
            if platform.system() == 'Windows':
                print(f"   ℹ️ PyTorch compile отключен для Windows (избегаем Triton)")
            else:
                # Проверяем, работает ли compile на Linux/Mac
                try:
                    test_model = torch.nn.Linear(10, 1)
                    compiled_model = torch.compile(test_model, mode='reduce-overhead')
                    test_input = torch.randn(1, 10)
                    _ = compiled_model(test_input)
                    print(f"   ✅ PyTorch compile работает")
                except Exception as e:
                    print(f"   ⚠️ PyTorch compile не работает: {e}")
                    print(f"   ℹ️ Это нормально - обучение будет работать без compile")
        else:
            print("\n8. PyTorch 2.0+ compile:")
            print(f"   ⚠️ PyTorch compile недоступен (нужен PyTorch 2.0+)")
        
        print("\n🎉 Все тесты прошли успешно!")
        print("✅ train.py исправлен и оптимизирован для RTX 4070 (Windows)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка в тесте: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_train_fix()
    if success:
        print("\n🚀 Можно запускать обучение с оптимизациями!")
        print("python train.py")
    else:
        print("\n⚠️ Нужно исправить ошибки перед запуском обучения") 