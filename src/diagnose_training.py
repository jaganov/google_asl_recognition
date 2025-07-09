# diagnose_training.py
import torch
import torch.nn as nn
import time
import psutil
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

def check_gpu_usage():
    """Проверка использования GPU"""
    print("🔍 ДИАГНОСТИКА GPU:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA недоступен!")
        return False
    
    print(f"✅ CUDA доступен")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Проверяем текущее использование памяти
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"📊 Начальная память GPU: {initial_memory:.2f} GB")
    
    return True

def test_data_loading_speed():
    """Тест скорости загрузки данных"""
    print("\n🔍 ТЕСТ ЗАГРУЗКИ ДАННЫХ:")
    print("=" * 50)
    
    try:
        from data_loader import ASLDataLoader
        
        print("📂 Загружаем DataLoader...")
        start_time = time.time()
        
        dataloader = ASLDataLoader(
            data_dir="../data/google_asl_signs",
            batch_size=12,
            max_len=384,
            num_workers=4
        )
        
        load_time = time.time() - start_time
        print(f"✅ DataLoader загружен за {load_time:.2f} сек")
        
        # Получаем dataloaders
        train_loader, val_loader, test_loader = dataloader.get_dataloaders(augment_train=True)
        
        print(f"✅ Train batches: {len(train_loader)}")
        print(f"✅ Val batches: {len(val_loader)}")
        print(f"✅ Test batches: {len(test_loader)}")
        
        # Тест скорости загрузки первого батча
        print("\n📊 Тестируем скорость загрузки батча...")
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Тестируем первые 3 батча
                break
            batch_time = time.time() - start_time
            print(f"   Батч {i+1}: {batch_time:.2f} сек")
            start_time = time.time()
            
            # Проверяем размеры данных
            features = batch['features']
            labels = batch['labels']
            print(f"   Размер features: {features.shape}")
            print(f"   Размер labels: {labels.shape}")
            print(f"   Тип features: {features.dtype}")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None, None, None

def test_gpu_transfer_speed(train_loader):
    """Тест скорости передачи данных на GPU"""
    print("\n🔍 ТЕСТ ПЕРЕДАЧИ НА GPU:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ GPU недоступен")
        return
    
    device = torch.device('cuda')
    
    # Очищаем память
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"📊 Память до передачи: {initial_memory:.2f} GB")
    
    # Тестируем передачу данных
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Тестируем первые 3 батча
            break
            
        start_time = time.time()
        
        # Передаем на GPU
        features = batch['features'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        transfer_time = time.time() - start_time
        
        # Проверяем память
        current_memory = torch.cuda.memory_allocated() / 1024**3
        cached_memory = torch.cuda.memory_reserved() / 1024**3
        
        print(f"   Батч {i+1}:")
        print(f"     Время передачи: {transfer_time:.3f} сек")
        print(f"     Память использована: {current_memory:.2f} GB")
        print(f"     Память зарезервирована: {cached_memory:.2f} GB")
        print(f"     Размер features на GPU: {features.shape}")
        
        # Очищаем для следующего теста
        del features, labels
        torch.cuda.empty_cache()

def test_model_forward_pass(train_loader):
    """Тест forward pass модели"""
    print("\n🔍 ТЕСТ FORWARD PASS:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ GPU недоступен")
        return
    
    device = torch.device('cuda')
    
    try:
        from models import get_model
        
        # Получаем размерность из первого батча
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['features'].shape[-1]
        
        print(f"📊 Размерность входных данных: {input_dim}")
        
        # Создаем модель
        model = get_model(
            input_dim=input_dim,
            num_classes=250,  # Примерное количество классов
            max_len=384,
            dim=192
        ).to(device)
        
        print(f"✅ Модель создана")
        print(f"📊 Параметры модели: {sum(p.numel() for p in model.parameters()):,}")
        
        # Тестируем forward pass
        model.train()
        torch.cuda.empty_cache()
        
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Тестируем первые 3 батча
                break
                
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Измеряем время forward pass
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():  # Для теста скорости
                outputs = model(features)
            
            torch.cuda.synchronize()
            forward_time = time.time() - start_time
            
            # Проверяем память
            current_memory = torch.cuda.memory_allocated() / 1024**3
            cached_memory = torch.cuda.memory_reserved() / 1024**3
            
            print(f"   Батч {i+1}:")
            print(f"     Forward pass: {forward_time:.3f} сек")
            print(f"     Память использована: {current_memory:.2f} GB")
            print(f"     Память зарезервирована: {cached_memory:.2f} GB")
            print(f"     Размер выходов: {outputs.shape}")
            
            # Очищаем
            del features, labels, outputs
            torch.cuda.empty_cache()
        
        return model
        
    except Exception as e:
        print(f"❌ Ошибка тестирования модели: {e}")
        return None

def test_full_training_step(model, train_loader):
    """Тест полного шага обучения"""
    print("\n🔍 ТЕСТ ПОЛНОГО ШАГА ОБУЧЕНИЯ:")
    print("=" * 50)
    
    if not torch.cuda.is_available() or model is None:
        print("❌ GPU или модель недоступны")
        return
    
    device = torch.device('cuda')
    
    # Настройки
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    model.train()
    torch.cuda.empty_cache()
    
    for i, batch in enumerate(train_loader):
        if i >= 2:  # Тестируем первые 2 батча
            break
            
        features = batch['features'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # Измеряем время полного шага
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
        current_memory = torch.cuda.memory_allocated() / 1024**3
        cached_memory = torch.cuda.memory_reserved() / 1024**3
        
        print(f"   Шаг {i+1}:")
        print(f"     Полное время: {step_time:.3f} сек")
        print(f"     Loss: {loss.item():.4f}")
        print(f"     Память использована: {current_memory:.2f} GB")
        print(f"     Память зарезервирована: {cached_memory:.2f} GB")
        
        # Очищаем
        del features, labels, outputs, loss
        torch.cuda.empty_cache()

def check_system_resources():
    """Проверка системных ресурсов"""
    print("\n🔍 СИСТЕМНЫЕ РЕСУРСЫ:")
    print("=" * 50)
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"📊 CPU: {cpu_count} ядер, использование: {cpu_percent:.1f}%")
    
    # RAM
    memory = psutil.virtual_memory()
    print(f"📊 RAM: {memory.total / 1024**3:.1f} GB всего, {memory.used / 1024**3:.1f} GB использовано ({memory.percent:.1f}%)")
    
    # Диск
    disk = psutil.disk_usage('/')
    print(f"📊 Диск: {disk.total / 1024**3:.1f} GB всего, {disk.used / 1024**3:.1f} GB использовано")
    
    # Проверяем наличие данных
    data_path = Path("../data/google_asl_signs")
    if data_path.exists():
        print(f"✅ Данные найдены: {data_path}")
        # Подсчитываем размер
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1
        print(f"📊 Файлов: {file_count:,}, Размер: {total_size / 1024**3:.1f} GB")
    else:
        print(f"❌ Данные не найдены: {data_path}")

def main():
    """Основная диагностика"""
    print("🔍 ДИАГНОСТИКА ПРОИЗВОДИТЕЛЬНОСТИ ОБУЧЕНИЯ")
    print("=" * 70)
    
    # 1. Проверка GPU
    gpu_ok = check_gpu_usage()
    
    # 2. Проверка системных ресурсов
    check_system_resources()
    
    # 3. Тест загрузки данных
    train_loader, val_loader, test_loader = test_data_loading_speed()
    
    if train_loader is not None:
        # 4. Тест передачи на GPU
        test_gpu_transfer_speed(train_loader)
        
        # 5. Тест forward pass
        model = test_model_forward_pass(train_loader)
        
        # 6. Тест полного шага обучения
        test_full_training_step(model, train_loader)
    
    print("\n🎯 РЕКОМЕНДАЦИИ:")
    print("=" * 50)
    
    if not gpu_ok:
        print("❌ Проблема: GPU недоступен")
        print("💡 Решение: Установите CUDA и PyTorch с поддержкой CUDA")
    else:
        print("✅ GPU работает корректно")
        
        if train_loader is None:
            print("❌ Проблема: Медленная загрузка данных")
            print("💡 Решения:")
            print("   - Уменьшите num_workers до 2")
            print("   - Отключите аугментации временно")
            print("   - Проверьте скорость диска")
        else:
            print("✅ Загрузка данных работает")
            print("💡 Для ускорения:")
            print("   - Используйте SSD для данных")
            print("   - Увеличьте batch_size если позволяет память")
            print("   - Включите mixed precision")

if __name__ == "__main__":
    main() 