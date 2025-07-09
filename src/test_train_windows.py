#!/usr/bin/env python3
"""
Максимально простой тест для Windows (без PyTorch compile)
"""

import torch
import sys
import os
import platform

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Подавляем ошибки Triton на Windows
if platform.system() == 'Windows':
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        print("🔧 Подавлены ошибки Triton для Windows")
    except:
        pass

# Настройка TensorFloat32
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    print("🔧 Включен TensorFloat32")

from train import ASLTrainer
from data_loader import ASLDataLoader
from preprocessing import ASLPreprocessor
from augmentations import ASLAugmentations

# Сброс флага для чистого теста
if hasattr(ASLPreprocessor, '_landmarks_printed'):
    delattr(ASLPreprocessor, '_landmarks_printed')

def test_windows_compatibility():
    """Тест совместимости с Windows"""
    print("🧪 Тест совместимости с Windows...")
    
    try:
        # 1. Проверка системы
        print(f"\n1. Система: {platform.system()}")
        print(f"   ✅ PyTorch версия: {torch.__version__}")
        print(f"   ✅ CUDA доступен: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        
        # 2. Тест аугментаций
        print("\n2. Тестируем аугментации...")
        augmenter = ASLAugmentations()
        test_features = torch.randn(2, 50, 468 * 3 * 3)
        augmented = augmenter(test_features)
        print(f"   ✅ Аугментации работают")
        
        # 3. Тест препроцессора
        print("\n3. Тестируем препроцессор...")
        preprocessor = ASLPreprocessor(max_len=32)  # Маленькая длина для теста
        test_landmarks = torch.randn(2, 20, 468, 3)
        features = preprocessor(test_landmarks)
        print(f"   ✅ Препроцессор работает")
        
        # 4. Тест DataLoader (используем тот же препроцессор)
        print("\n4. Тестируем DataLoader...")
        dataloader = ASLDataLoader(
            data_dir="../data/google_asl_signs",
            batch_size=2,  # Очень маленький batch для теста
            max_len=32,
            preprocessor=preprocessor  # Передаем существующий препроцессор
        )
        
        train_loader, val_loader, test_loader = dataloader.get_dataloaders(
            augment_train=True
        )
        
        sample_batch = next(iter(train_loader))
        print(f"   ✅ DataLoader работает")
        
        # 5. Тест тренера (без compile)
        print("\n5. Тестируем тренер (без PyTorch compile)...")
        trainer = ASLTrainer(
            data_dir="../data/google_asl_signs",
            model_dir="models/test_windows",
            max_len=32,
            batch_size=2,
            dim=32,  # Очень маленькая модель
            lr=1e-3,
            epochs=1,
            use_augmentations=True,
            use_mixed_precision=False,  # Отключаем для теста
            gradient_clip_val=1.0,
            gradient_accumulation_steps=1
        )
        
        print(f"   ✅ Тренер создан")
        print(f"   ✅ Параметры модели: {sum(p.numel() for p in trainer.model.parameters()):,}")
        
        # 6. Тест одного батча обучения
        print("\n6. Тестируем один батч обучения...")
        trainer.model.train()
        
        # Берем один батч
        batch = next(iter(trainer.train_loader))
        features = batch['features'].to(trainer.device)
        labels = batch['labels'].to(trainer.device)
        
        # Forward pass
        outputs = trainer.model(features)
        loss = trainer.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        print(f"   ✅ Forward/backward pass работает")
        print(f"   ✅ Loss: {loss.item():.4f}")
        
        # 7. Тест валидации
        print("\n7. Тестируем валидацию...")
        trainer.model.eval()
        with torch.no_grad():
            val_batch = next(iter(trainer.val_loader))
            val_features = val_batch['features'].to(trainer.device)
            val_labels = val_batch['labels'].to(trainer.device)
            val_outputs = trainer.model(val_features)
            val_loss = trainer.criterion(val_outputs, val_labels)
        
        print(f"   ✅ Валидация работает")
        print(f"   ✅ Val Loss: {val_loss.item():.4f}")
        
        print("\n🎉 Все тесты прошли успешно!")
        print("✅ Система полностью совместима с Windows")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_windows_compatibility()
    if success:
        print("\n🚀 Можно запускать обучение!")
        print("python train.py")
    else:
        print("\n⚠️ Нужно исправить ошибки") 