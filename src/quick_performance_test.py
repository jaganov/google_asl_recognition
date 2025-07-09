# quick_performance_test.py
import torch
import time
import psutil
import os
from pathlib import Path

def quick_gpu_test():
    """Быстрый тест GPU"""
    print("🔍 БЫСТРЫЙ ТЕСТ GPU:")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA недоступен!")
        return False
    
    print(f"✅ CUDA доступен")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Тест памяти
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"📊 Начальная память: {initial_memory:.2f} GB")
    
    # Создаем тестовый тензор
    test_tensor = torch.randn(1000, 1000).cuda()
    memory_after = torch.cuda.memory_allocated() / 1024**3
    print(f"📊 Память после создания тензора: {memory_after:.2f} GB")
    
    # Тест вычислений
    start_time = time.time()
    result = torch.mm(test_tensor, test_tensor)
    torch.cuda.synchronize()
    compute_time = time.time() - start_time
    print(f"⏱️ Время матричного умножения: {compute_time:.3f} сек")
    
    # Очищаем
    del test_tensor, result
    torch.cuda.empty_cache()
    
    return True

def quick_data_loading_test():
    """Быстрый тест загрузки данных"""
    print("\n🔍 БЫСТРЫЙ ТЕСТ ЗАГРУЗКИ ДАННЫХ:")
    print("=" * 40)
    
    try:
        from data_loader import ASLDataLoader
        
        print("📂 Создаем DataLoader...")
        start_time = time.time()
        
        dataloader = ASLDataLoader(
            data_dir="../data/google_asl_signs",
            batch_size=8,  # Маленький batch для быстрого теста
            max_len=128,   # Короткая последовательность
            num_workers=1  # Один воркер для теста
        )
        
        load_time = time.time() - start_time
        print(f"✅ DataLoader создан за {load_time:.2f} сек")
        
        # Получаем dataloaders
        print("📂 Получаем dataloaders...")
        start_time = time.time()
        train_loader, val_loader, test_loader = dataloader.get_dataloaders(augment_train=False)
        dataloader_time = time.time() - start_time
        print(f"✅ DataLoader'ы созданы за {dataloader_time:.2f} сек")
        
        print(f"📊 Train batches: {len(train_loader)}")
        print(f"📊 Val batches: {len(val_loader)}")
        print(f"📊 Test batches: {len(test_loader)}")
        
        # Тест загрузки первого батча
        print("\n📂 Тестируем загрузку первого батча...")
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            if i >= 2:  # Только первые 2 батча
                break
            batch_time = time.time() - start_time
            print(f"   Батч {i+1}: {batch_time:.2f} сек")
            print(f"   Размер: {batch['features'].shape}")
            start_time = time.time()
        
        return train_loader
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None

def quick_model_test(train_loader):
    """Быстрый тест модели"""
    print("\n🔍 БЫСТРЫЙ ТЕСТ МОДЕЛИ:")
    print("=" * 40)
    
    if not torch.cuda.is_available() or train_loader is None:
        print("❌ GPU или данные недоступны")
        return None
    
    try:
        from models import get_model
        
        device = torch.device('cuda')
        
        # Получаем размерность из первого батча
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['features'].shape[-1]
        
        print(f"📊 Размерность входных данных: {input_dim}")
        
        # Создаем модель
        print("🤖 Создаем модель...")
        start_time = time.time()
        model = get_model(
            input_dim=input_dim,
            num_classes=250,
            max_len=128,  # Короткая последовательность для теста
            dim=128       # Меньшая модель для теста
        ).to(device)
        model_time = time.time() - start_time
        print(f"✅ Модель создана за {model_time:.2f} сек")
        print(f"📊 Параметры: {sum(p.numel() for p in model.parameters()):,}")
        
        # Тест forward pass
        model.train()
        torch.cuda.empty_cache()
        
        print("\n📊 Тестируем forward pass...")
        for i, batch in enumerate(train_loader):
            if i >= 2:  # Только первые 2 батча
                break
                
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Измеряем время
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(features)
            
            torch.cuda.synchronize()
            forward_time = time.time() - start_time
            
            # Проверяем память
            memory_used = torch.cuda.memory_allocated() / 1024**3
            
            print(f"   Батч {i+1}:")
            print(f"     Forward pass: {forward_time:.3f} сек")
            print(f"     Память: {memory_used:.2f} GB")
            print(f"     Размер выходов: {outputs.shape}")
            
            # Очищаем
            del features, labels, outputs
            torch.cuda.empty_cache()
        
        return model
        
    except Exception as e:
        print(f"❌ Ошибка тестирования модели: {e}")
        return None

def quick_training_test(model, train_loader):
    """Быстрый тест обучения"""
    print("\n🔍 БЫСТРЫЙ ТЕСТ ОБУЧЕНИЯ:")
    print("=" * 40)
    
    if not torch.cuda.is_available() or model is None or train_loader is None:
        print("❌ GPU, модель или данные недоступны")
        return
    
    device = torch.device('cuda')
    
    # Настройки
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    torch.cuda.empty_cache()
    
    print("📊 Тестируем полный шаг обучения...")
    for i, batch in enumerate(train_loader):
        if i >= 2:  # Только первые 2 батча
            break
            
        features = batch['features'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # Измеряем время
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        step_time = time.time() - start_time
        
        # Проверяем память
        memory_used = torch.cuda.memory_allocated() / 1024**3
        
        print(f"   Шаг {i+1}:")
        print(f"     Полное время: {step_time:.3f} сек")
        print(f"     Loss: {loss.item():.4f}")
        print(f"     Память: {memory_used:.2f} GB")
        
        # Очищаем
        del features, labels, outputs, loss
        torch.cuda.empty_cache()

def check_system_resources():
    """Проверка системных ресурсов"""
    print("\n🔍 СИСТЕМНЫЕ РЕСУРСЫ:")
    print("=" * 40)
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"📊 CPU: {cpu_count} ядер, использование: {cpu_percent:.1f}%")
    
    # RAM
    memory = psutil.virtual_memory()
    print(f"📊 RAM: {memory.total / 1024**3:.1f} GB всего, {memory.used / 1024**3:.1f} GB использовано ({memory.percent:.1f}%)")
    
    # Диск
    try:
        disk = psutil.disk_usage('/')
        print(f"📊 Диск: {disk.total / 1024**3:.1f} GB всего, {disk.used / 1024**3:.1f} GB использовано")
    except:
        print("📊 Диск: информация недоступна")
    
    # Проверяем наличие данных
    data_path = Path("../data/google_asl_signs")
    if data_path.exists():
        print(f"✅ Данные найдены: {data_path}")
        # Быстрая проверка размера
        try:
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(data_path):
                for file in files[:100]:  # Только первые 100 файлов для скорости
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                break  # Только первый уровень
            print(f"📊 Примерный размер (первые 100 файлов): {total_size / 1024**3:.1f} GB")
        except:
            print("📊 Размер данных: не удалось определить")
    else:
        print(f"❌ Данные не найдены: {data_path}")

def main():
    """Основной быстрый тест"""
    print("🔍 БЫСТРАЯ ДИАГНОСТИКА ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    # 1. Проверка GPU
    gpu_ok = quick_gpu_test()
    
    # 2. Проверка системных ресурсов
    check_system_resources()
    
    # 3. Тест загрузки данных
    train_loader = quick_data_loading_test()
    
    if train_loader is not None:
        # 4. Тест модели
        model = quick_model_test(train_loader)
        
        # 5. Тест обучения
        quick_training_test(model, train_loader)
    
    print("\n🎯 РЕКОМЕНДАЦИИ:")
    print("=" * 40)
    
    if not gpu_ok:
        print("❌ Проблема: GPU недоступен")
        print("💡 Решение: Установите CUDA и PyTorch с поддержкой CUDA")
    else:
        print("✅ GPU работает корректно")
        
        if train_loader is None:
            print("❌ Проблема: Медленная загрузка данных")
            print("💡 Решения:")
            print("   - Уменьшите num_workers до 1")
            print("   - Отключите аугментации")
            print("   - Проверьте скорость диска")
        else:
            print("✅ Загрузка данных работает")
            print("💡 Для ускорения:")
            print("   - Используйте SSD для данных")
            print("   - Увеличьте batch_size")
            print("   - Включите mixed precision")

if __name__ == "__main__":
    main() 