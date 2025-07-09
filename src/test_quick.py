#!/usr/bin/env python3
"""
Максимально быстрый тест без лишних препроцессоров
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

# Сброс флага для чистого теста
from preprocessing import ASLPreprocessor
if hasattr(ASLPreprocessor, '_landmarks_printed'):
    delattr(ASLPreprocessor, '_landmarks_printed')

from train import ASLTrainer

def test_quick():
    """Максимально быстрый тест"""
    print("🧪 Быстрый тест совместимости...")
    
    try:
        # 1. Проверка системы
        print(f"\n1. Система: {platform.system()}")
        print(f"   ✅ PyTorch версия: {torch.__version__}")
        print(f"   ✅ CUDA доступен: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        
        # 2. Тест тренера (создает только один препроцессор)
        print("\n2. Тестируем тренер...")
        trainer = ASLTrainer(
            data_dir="../data/google_asl_signs",
            model_dir="models/test_quick",
            max_len=32,
            batch_size=2,
            dim=32,
            lr=1e-3,
            epochs=1,
            use_augmentations=True,
            use_mixed_precision=False,  # Отключаем для простоты
            gradient_clip_val=1.0,
            gradient_accumulation_steps=1
        )
        
        print(f"   ✅ Тренер создан")
        print(f"   ✅ Параметры модели: {sum(p.numel() for p in trainer.model.parameters()):,}")
        
        # 3. Тест одного батча
        print("\n3. Тестируем один батч...")
        trainer.model.train()
        
        batch = next(iter(trainer.train_loader))
        features = batch['features'].to(trainer.device)
        labels = batch['labels'].to(trainer.device)
        
        outputs = trainer.model(features)
        loss = trainer.criterion(outputs, labels)
        
        print(f"   ✅ Forward pass работает")
        print(f"   ✅ Loss: {loss.item():.4f}")
        
        # 4. Тест валидации
        print("\n4. Тестируем валидацию...")
        trainer.model.eval()
        with torch.no_grad():
            val_batch = next(iter(trainer.val_loader))
            val_features = val_batch['features'].to(trainer.device)
            val_labels = val_batch['labels'].to(trainer.device)
            val_outputs = trainer.model(val_features)
            val_loss = trainer.criterion(val_outputs, val_labels)
        
        print(f"   ✅ Валидация работает")
        print(f"   ✅ Val Loss: {val_loss.item():.4f}")
        
        print("\n🎉 Быстрый тест прошел успешно!")
        print("✅ Система готова к обучению")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quick()
    if success:
        print("\n🚀 Можно запускать обучение!")
        print("python train.py")
    else:
        print("\n⚠️ Нужно исправить ошибки") 