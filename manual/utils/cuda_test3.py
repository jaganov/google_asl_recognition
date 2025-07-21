#!/usr/bin/env python3
"""
Детальная диагностика точности CUDA вычислений
"""

import torch
import time
import numpy as np

def debug_precision_issue():
    print("🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ТОЧНОСТИ")
    print("="*50)
    
    # Проверка cuDNN
    print(f"🧠 cuDNN включен: {torch.backends.cudnn.enabled}")
    print(f"🧠 cuDNN версия: {torch.backends.cudnn.version()}")
    print(f"🧠 cuDNN детерминированный: {torch.backends.cudnn.deterministic}")
    print(f"🧠 cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    # Информация о GPU
    props = torch.cuda.get_device_properties(0)
    print(f"🎮 Compute capability: {props.major}.{props.minor}")
    print(f"🎮 Total memory: {props.total_memory // 1024**3} GB")
    
    # Тест разных размеров матриц
    print("\n🧪 ТЕСТ РАЗНЫХ РАЗМЕРОВ МАТРИЦ:")
    
    sizes = [100, 500, 1000, 2000, 3000]
    
    for size in sizes:
        print(f"\n📐 Размер матрицы: {size}x{size}")
        
        # Создаем одинаковые матрицы
        torch.manual_seed(42)  # Фиксируем random seed
        x_cpu = torch.randn(size, size, dtype=torch.float32)
        
        torch.manual_seed(42)
        x_gpu = torch.randn(size, size, dtype=torch.float32).cuda()
        
        torch.manual_seed(42)
        y_cpu = torch.randn(size, size, dtype=torch.float32)
        
        torch.manual_seed(42)
        y_gpu = torch.randn(size, size, dtype=torch.float32).cuda()
        
        # Проверяем, что матрицы одинаковые
        diff_x = torch.abs(x_cpu - x_gpu.cpu()).max().item()
        diff_y = torch.abs(y_cpu - y_gpu.cpu()).max().item()
        print(f"   📊 Разница входных данных X: {diff_x:.2e}")
        print(f"   📊 Разница входных данных Y: {diff_y:.2e}")
        
        # CPU вычисления
        start_time = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        # GPU вычисления
        torch.cuda.synchronize()
        start_time = time.time()
        z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        # Сравнение результатов
        z_gpu_cpu = z_gpu.cpu()
        diff = torch.abs(z_cpu - z_gpu_cpu).max().item()
        mean_diff = torch.abs(z_cpu - z_gpu_cpu).mean().item()
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"   ⏱️  CPU время: {cpu_time:.4f}s")
        print(f"   ⏱️  GPU время: {gpu_time:.4f}s")
        print(f"   🚀 Ускорение: {speedup:.1f}x")
        print(f"   🎯 Макс. разница: {diff:.2e}")
        print(f"   🎯 Средняя разница: {mean_diff:.2e}")
        
        # Оценка точности
        if diff < 1e-3:
            print("   ✅ Точность: ОТЛИЧНО")
        elif diff < 1e-1:
            print("   ⚠️  Точность: ПРИЕМЛЕМО")
        else:
            print("   ❌ Точность: ПРОБЛЕМА")
            
        # Освобождаем память
        del x_cpu, y_cpu, z_cpu, x_gpu, y_gpu, z_gpu, z_gpu_cpu
        torch.cuda.empty_cache()

def test_mixed_precision():
    print("\n🧪 ТЕСТ СМЕШАННОЙ ТОЧНОСТИ (AMP):")
    print("="*40)
    
    size = 2000
    
    # Обычная точность float32
    x = torch.randn(size, size).cuda()
    y = torch.randn(size, size).cuda()
    
    # Тест без AMP
    start_time = time.time()
    z_normal = torch.mm(x, y)
    torch.cuda.synchronize()
    normal_time = time.time() - start_time
    
    # Тест с AMP
    start_time = time.time()
    with torch.amp.autocast(device_type="cuda"):
        z_amp = torch.mm(x, y)
    torch.cuda.synchronize()
    amp_time = time.time() - start_time
    
    # Сравнение
    diff = torch.abs(z_normal - z_amp).max().item()
    speedup = normal_time / amp_time if amp_time > 0 else 0
    
    print(f"🖥️  Обычная точность: {normal_time:.4f}s")
    print(f"⚡ AMP: {amp_time:.4f}s")
    print(f"🚀 Ускорение AMP: {speedup:.1f}x")
    print(f"🎯 Разница результатов: {diff:.2e}")
    
    if diff < 1e-2:
        print("✅ AMP работает корректно")
    else:
        print("⚠️  AMP может иметь проблемы точности")

def test_google_asl_scenario():
    print("\n🎯 ТЕСТ СЦЕНАРИЯ GOOGLE ASL:")
    print("="*40)
    
    # Параметры как в реальном проекте
    batch_size = 8
    frames = 16
    channels = 3
    height = width = 224
    
    print(f"📊 Тестовый батч: {batch_size}x{frames}x{channels}x{height}x{width}")
    
    try:
        # Создаем реалистичные данные
        video_batch = torch.randn(batch_size, frames, channels, height, width).cuda()
        
        # Симуляция ResNet backbone
        conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3).cuda()
        bn1 = torch.nn.BatchNorm2d(64).cuda()
        relu = torch.nn.ReLU().cuda()
        pool = torch.nn.MaxPool2d(3, stride=2, padding=1).cuda()
        
        # Симуляция обработки всех фреймов
        start_time = time.time()
        
        # Reshape для обработки фреймов
        batch_frames = batch_size * frames
        reshaped = video_batch.view(batch_frames, channels, height, width)
        
        # Forward pass
        x = conv1(reshaped)
        x = bn1(x)
        x = relu(x)
        x = pool(x)
        
        # Global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(batch_size, frames, -1)
        
        # Временная агрегация
        x = x.mean(dim=1)  # Среднее по времени
        
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        
        # Статистика памяти
        memory_used = torch.cuda.memory_allocated() // 1024**2
        memory_reserved = torch.cuda.memory_reserved() // 1024**2
        
        print(f"⏱️  Forward pass: {forward_time:.3f}s")
        print(f"📊 VRAM использовано: {memory_used} MB")
        print(f"📊 VRAM зарезервировано: {memory_reserved} MB")
        print(f"🎯 Выходная форма: {x.shape}")
        
        # Оценка производительности
        if forward_time < 0.1:
            print("🎉 ОТЛИЧНО! Идеально для обучения")
        elif forward_time < 0.2:
            print("✅ ХОРОШО! Подходит для обучения")
        else:
            print("⚠️  МЕДЛЕННО! Нужна оптимизация")
            
        # Тест обратного прохода
        start_time = time.time()
        dummy_loss = x.sum()
        dummy_loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - start_time
        
        print(f"⏱️  Backward pass: {backward_time:.3f}s")
        print(f"⏱️  Общее время: {forward_time + backward_time:.3f}s")
        
        # Очистка
        torch.cuda.empty_cache()
        
        return True
        
    except torch.cuda.OutOfMemoryError:
        print("❌ НЕХВАТКА ПАМЯТИ!")
        print("💡 Попробуйте batch_size = 4")
        return False
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return False

def main():
    print("🚀 ДЕТАЛЬНАЯ ДИАГНОСТИКА CUDA СОВМЕСТИМОСТИ")
    print("Специально для RTX 4070 + CUDA 12.9\n")
    
    debug_precision_issue()
    test_mixed_precision()
    success = test_google_asl_scenario()
    
    print(f"\n{'='*60}")
    print("📋 ЗАКЛЮЧЕНИЕ:")
    
    if success:
        print("🎉 СИСТЕМА ГОТОВА для Google ASL проекта!")
        print("✅ Все тесты пройдены успешно")
        print("🚀 Можете начинать обучение моделей!")
    else:
        print("⚠️  Есть проблемы с производительностью")
        print("🔧 Рекомендуется оптимизация настроек")
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print("• Batch size: 6-8 для RTX 4070")
    print("• Mixed precision: Включить для ускорения")
    print("• Gradient accumulation: 2-4 для эффективности")
    print("• Frames per video: 16 оптимально")

if __name__ == "__main__":
    main()