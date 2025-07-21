#!/usr/bin/env python3
"""
Быстрый тест совместимости CUDA - ДО и ПОСЛЕ переустановки PyTorch
"""

import torch
import time
import subprocess

def test_cuda_compatibility():
    print("🔍 ТЕСТ СОВМЕСТИМОСТИ CUDA")
    print("="*50)
    
    # Информация о системе
    print(f"🔗 PyTorch версия: {torch.__version__}")
    print(f"🔥 PyTorch CUDA версия: {torch.version.cuda}")
    
    # Проверка доступности CUDA
    cuda_available = torch.cuda.is_available()
    print(f"⚡ CUDA доступна: {'✅ ДА' if cuda_available else '❌ НЕТ'}")
    
    if not cuda_available:
        print("\n🚨 ПРОБЛЕМА: PyTorch не видит CUDA!")
        print("Возможные причины:")
        print("   • Несовместимость версий CUDA")
        print("   • Драйвер NVIDIA устарел")
        print("   • PyTorch собран без CUDA поддержки")
        return False
    
    # Информация о GPU
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Проверка драйвера через nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'CUDA Version' in line:
                driver_cuda = line.split('CUDA Version: ')[1].split()[0]
                print(f"🏗️  Драйвер поддерживает CUDA: {driver_cuda}")
                break
    except:
        print("⚠️  Не удалось получить версию драйвера")
    
    # КРИТИЧЕСКИЙ ТЕСТ: Скорость вычислений
    print("\n🧪 ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ:")
    
    # CPU тест
    x_cpu = torch.randn(2000, 2000)
    y_cpu = torch.randn(2000, 2000)
    
    start_time = time.time()
    z_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start_time
    print(f"🖥️  CPU время: {cpu_time:.3f}s")
    
    # GPU тест
    try:
        x_gpu = torch.randn(2000, 2000).cuda()
        y_gpu = torch.randn(2000, 2000).cuda()
        
        # Прогрев GPU
        torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"🎮 GPU время: {gpu_time:.3f}s")
        
        speedup = cpu_time / gpu_time
        print(f"🚀 Ускорение: {speedup:.1f}x")
        
        # Проверка корректности
        z_gpu_cpu = z_gpu.cpu()
        diff = torch.abs(z_cpu - z_gpu_cpu).max().item()
        print(f"🎯 Точность: {diff:.2e} (должно быть < 1e-3)")
        
        # Использование памяти
        memory_used = torch.cuda.memory_allocated() // 1024**2
        print(f"📊 VRAM использовано: {memory_used} MB")
        
        # Оценка результатов
        if speedup > 5 and diff < 1e-3:
            print("\n🎉 ОТЛИЧНО! CUDA работает правильно!")
            print(f"✅ RTX 4070 дает {speedup:.1f}x ускорение")
            return True
        elif speedup > 1:
            print(f"\n⚠️  CUDA работает, но медленно ({speedup:.1f}x)")
            print("Возможно есть проблемы совместимости")
            return False
        else:
            print(f"\n❌ ПРОБЛЕМА! GPU медленнее CPU!")
            print("Скорее всего вычисления идут на CPU")
            return False
            
    except Exception as e:
        print(f"\n❌ ОШИБКА GPU теста: {e}")
        print("Вероятно проблемы с CUDA совместимостью")
        return False

def check_google_asl_readiness():
    print("\n🎯 ГОТОВНОСТЬ К GOOGLE ASL ПРОЕКТУ:")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ Не готов: CUDA недоступна")
        return
    
    try:
        # Тест батча для Google ASL (как в реальном проекте)
        batch_size = 8
        frames = 16
        channels = 3
        height = width = 224
        
        print(f"🧪 Тестируем батч: {batch_size}x{frames}x{channels}x{height}x{width}")
        
        # Создаем тестовый батч видео
        video_batch = torch.randn(batch_size, frames, channels, height, width).cuda()
        
        # Симулируем forward pass CNN
        start_time = time.time()
        
        # Flatten frames для CNN
        reshaped = video_batch.view(-1, channels, height, width)
        
        # Простая симуляция CNN слоев
        conv1 = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
        conv2 = torch.nn.Conv2d(64, 128, 3, padding=1).cuda()
        pool = torch.nn.AdaptiveAvgPool2d(1).cuda()
        
        with torch.no_grad():
            x = torch.relu(conv1(reshaped))
            x = torch.relu(conv2(x))
            x = pool(x)
        
        forward_time = time.time() - start_time
        
        # Проверка памяти
        memory_used = torch.cuda.memory_allocated() // 1024**2
        memory_total = torch.cuda.get_device_properties(0).total_memory // 1024**2
        memory_percent = memory_used / memory_total * 100
        
        print(f"⏱️  Forward pass: {forward_time:.3f}s")
        print(f"📊 Память: {memory_used}MB / {memory_total}MB ({memory_percent:.1f}%)")
        
        if forward_time < 0.1 and memory_percent < 50:
            print("🎉 ОТЛИЧНО! Готов к Google ASL проекту!")
            print(f"✅ Можно использовать батч размер {batch_size}")
        elif forward_time < 0.2:
            print("✅ Хорошо! Готов к проекту")
            print("💡 Возможно стоит уменьшить батч до 4-6")
        else:
            print("⚠️  Медленно для продуктивной работы")
            print("🔧 Нужна оптимизация или меньший батч")
            
    except torch.cuda.OutOfMemoryError:
        print("❌ Нехватка VRAM для батча размер 8")
        print("💡 Попробуйте батч размер 2-4")
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")

if __name__ == "__main__":
    print("🚀 ДИАГНОСТИКА СОВМЕСТИМОСТИ CUDA")
    print("Запустите этот тест ДО и ПОСЛЕ переустановки PyTorch\n")
    
    success = test_cuda_compatibility()
    
    if success:
        check_google_asl_readiness()
    
    print(f"\n{'='*50}")
    if success:
        print("🎯 РЕЗУЛЬТАТ: Система работает правильно!")
    else:
        print("🔧 РЕЗУЛЬТАТ: Нужна переустановка PyTorch!")
        print("💡 Команда: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")