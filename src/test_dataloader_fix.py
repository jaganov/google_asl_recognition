#!/usr/bin/env python3
"""
Тест исправления data_loader
"""

import torch
from data_loader import ASLDataset, ASLDataLoader
from preprocessing import ASLPreprocessor

def test_dataloader_fix():
    """Тест исправления data_loader"""
    print("🧪 Тестируем исправление data_loader...")
    
    try:
        # Создаем препроцессор
        preprocessor = ASLPreprocessor(max_len=64)
        
        # Создаем dataset
        dataset = ASLDataset(
            data_dir="../data/google_asl_signs",
            split_file="splits/train.csv",
            preprocessor=preprocessor,
            max_len=64,
            augment=True
        )
        
        print(f"   Dataset создан: {len(dataset)} образцов")
        
        # Тестируем один образец
        sample = dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Features shape: {sample['landmarks'].shape}")  # Теперь это features
        print(f"   Label: {sample['label']}")
        
        # Тестируем DataLoader
        dataloader = ASLDataLoader(batch_size=2, max_len=64)
        train_loader, val_loader, test_loader = dataloader.get_dataloaders()
        
        # Тестируем один батч
        batch = next(iter(train_loader))
        print(f"   Batch keys: {list(batch.keys())}")
        print(f"   Features shape: {batch['features'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")
        print(f"   Lengths shape: {batch['lengths'].shape}")
        
        print("✅ DataLoader работает корректно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в data_loader: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataloader_fix()
    if success:
        print("\n🎉 Исправление успешно!")
    else:
        print("\n⚠️ Требуется дополнительная отладка") 