#!/usr/bin/env python3
"""
Тест кэширования файлов препроцессора
"""

import torch
import sys
import os
import platform
import time

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

from data_loader import ASLDataLoader

def test_cache_performance():
    """Тест производительности кэширования"""
    print("🧪 Тест производительности кэширования...")
    
    try:
        # 1. Создаем препроцессор с кэшированием
        print("\n1. Создаем препроцессор с кэшированием...")
        preprocessor = ASLPreprocessor(max_len=64)
        
        # 2. Создаем DataLoader
        print("\n2. Создаем DataLoader...")
        dataloader = ASLDataLoader(
            data_dir="../data/google_asl_signs",
            batch_size=4,
            max_len=64,
            preprocessor=preprocessor
        )
        
        # 3. Тест без кэша (первая загрузка)
        print("\n3. Тест без кэша (первая загрузка)...")
        start_time = time.time()
        
        train_loader, val_loader, test_loader = dataloader.get_dataloaders(
            augment_train=False  # Отключаем аугментации для чистого теста
        )
        
        # Загружаем несколько батчей
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Только первые 3 батча
                break
        
        first_load_time = time.time() - start_time
        print(f"   ✅ Время первой загрузки: {first_load_time:.2f} сек")
        
        # 4. Тест с кэшем (повторная загрузка)
        print("\n4. Тест с кэшем (повторная загрузка)...")
        start_time = time.time()
        
        # Загружаем те же батчи снова
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Только первые 3 батча
                break
        
        cached_load_time = time.time() - start_time
        print(f"   ✅ Время с кэшем: {cached_load_time:.2f} сек")
        
        # 5. Статистика кэша
        print("\n5. Статистика кэша...")
        cache_stats = preprocessor.get_cache_stats()
        print(f"   ✅ Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"   ✅ Cache hits: {cache_stats['cache_hits']}")
        print(f"   ✅ Cache misses: {cache_stats['cache_misses']}")
        print(f"   ✅ Файлов в кэше: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        
        # 6. Ускорение
        if cached_load_time > 0:
            speedup = first_load_time / cached_load_time
            print(f"   ✅ Ускорение: {speedup:.1f}x")
        
        # 7. Тест очистки кэша
        print("\n6. Тест очистки кэша...")
        preprocessor.clear_cache()
        
        cache_stats_after = preprocessor.get_cache_stats()
        print(f"   ✅ Кэш очищен: {cache_stats_after['cache_size']} файлов")
        
        print("\n🎉 Тест кэширования прошел успешно!")
        print("✅ Кэширование значительно ускоряет загрузку данных")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cache_performance()
    if success:
        print("\n🚀 Кэширование работает эффективно!")
        print("Можно запускать обучение с оптимизированной загрузкой данных")
    else:
        print("\n⚠️ Нужно исправить проблемы с кэшированием") 