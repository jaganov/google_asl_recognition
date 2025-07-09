#!/usr/bin/env python3
"""
Минимальный тест без загрузки данных
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

from models import get_model

def test_minimal():
    """Минимальный тест без загрузки данных"""
    print("🧪 Минимальный тест совместимости...")
    
    try:
        # 1. Проверка системы
        print(f"\n1. Система: {platform.system()}")
        print(f"   ✅ PyTorch версия: {torch.__version__}")
        print(f"   ✅ CUDA доступен: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        
        # 2. Тест препроцессора (без загрузки файлов)
        print("\n2. Тестируем препроцессор...")
        preprocessor = ASLPreprocessor(max_len=32)
        test_landmarks = torch.randn(2, 20, 468, 3)  # Синтетические данные
        features = preprocessor(test_landmarks)
        print(f"   ✅ Препроцессор работает: {test_landmarks.shape} -> {features.shape}")
        
        # 3. Тест модели (без загрузки данных)
        print("\n3. Тестируем модель...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = get_model(
            input_dim=features.shape[-1],
            num_classes=250,  # Примерное количество классов
            max_len=32,
            dim=32
        ).to(device)
        
        print(f"   ✅ Модель создана")
        print(f"   ✅ Параметры модели: {sum(p.numel() for p in model.parameters()):,}")
        
        # 4. Тест forward pass
        print("\n4. Тестируем forward pass...")
        model.train()
        
        # Синтетические данные
        batch_features = torch.randn(2, 20, features.shape[-1]).to(device)
        batch_labels = torch.randint(0, 250, (2,)).to(device)
        
        outputs = model(batch_features)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, batch_labels)
        
        print(f"   ✅ Forward pass работает")
        print(f"   ✅ Loss: {loss.item():.4f}")
        print(f"   ✅ Output shape: {outputs.shape}")
        
        # 5. Тест backward pass
        print("\n5. Тестируем backward pass...")
        loss.backward()
        print(f"   ✅ Backward pass работает")
        
        print("\n🎉 Минимальный тест прошел успешно!")
        print("✅ Базовая функциональность работает")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal()
    if success:
        print("\n🚀 Базовая функциональность работает!")
        print("Можно запускать полные тесты:")
        print("python test_quick.py")
    else:
        print("\n⚠️ Нужно исправить базовые ошибки") 