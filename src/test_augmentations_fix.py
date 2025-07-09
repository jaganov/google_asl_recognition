#!/usr/bin/env python3
"""
Тест исправления аугментаций
"""

import torch
from augmentations import ASLAugmentations

def test_augmentations_fix():
    """Тест исправления аугментаций"""
    print("🧪 Тестируем исправление аугментаций...")
    
    # Тестовые данные
    batch_size, frames, features = 2, 50, 468 * 3 * 3
    x = torch.randn(batch_size, frames, features)
    
    augmenter = ASLAugmentations()
    
    print(f"   Исходная форма: {x.shape}")
    
    # Тест temporal_resample
    x_temp = augmenter.temporal_resample(x.clone())
    print(f"   Temporal resample: {x_temp.shape} - размер сохранен: {x_temp.shape == x.shape}")
    
    # Тест всех аугментаций
    x_aug = augmenter(x.clone())
    print(f"   Все аугментации: {x_aug.shape} - размер сохранен: {x_aug.shape == x.shape}")
    
    if x_aug.shape == x.shape:
        change_mag = torch.mean(torch.abs(x_aug - x))
        print(f"   Величина изменений: {change_mag:.6f}")
        print("✅ Аугментации работают корректно!")
        return True
    else:
        print(f"❌ Размеры не совпадают: {x.shape} -> {x_aug.shape}")
        return False

if __name__ == "__main__":
    success = test_augmentations_fix()
    if success:
        print("\n🎉 Исправление успешно!")
    else:
        print("\n⚠️ Требуется дополнительная отладка") 