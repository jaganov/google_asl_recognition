#!/usr/bin/env python3
"""
🔥 GPU Stress Test с встроенным мониторингом для RTX 4070
Показывает использование GPU без nvidia-smi pmon
"""

import torch
import time
import subprocess
import threading
import os
import json
from datetime import datetime

class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.stats = []
    
    def get_gpu_stats(self):
        """Получить статистику GPU через nvidia-smi"""
        try:
            cmd = [
                'nvidia-smi', 
                '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                line = result.stdout.strip()
                values = [v.strip() for v in line.split(',')]
                
                return {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'name': values[0],
                    'memory_used': int(values[1]),
                    'memory_total': int(values[2]),
                    'gpu_util': int(values[3]),
                    'temperature': int(values[4]),
                    'power_draw': float(values[5])
                }
        except Exception as e:
            print(f"❌ Ошибка мониторинга: {e}")
        return None
    
    def start_monitoring(self):
        """Запустить мониторинг в отдельном потоке"""
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                stats = self.get_gpu_stats()
                if stats:
                    self.stats.append(stats)
                    self.print_stats(stats)
                time.sleep(2)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
    
    def stop_monitoring(self):
        """Остановить мониторинг"""
        self.monitoring = False
    
    def print_stats(self, stats):
        """Красивый вывод статистики"""
        memory_percent = (stats['memory_used'] / stats['memory_total']) * 100
        
        print(f"\n📊 [{stats['timestamp']}] GPU Status:")
        print(f"   🎮 {stats['name']}")
        print(f"   💾 Memory: {stats['memory_used']}MB / {stats['memory_total']}MB ({memory_percent:.1f}%)")
        print(f"   🔥 GPU Util: {stats['gpu_util']}%")
        print(f"   🌡️  Temp: {stats['temperature']}°C")
        print(f"   ⚡ Power: {stats['power_draw']}W")
        print("   " + "="*50)

def get_process_info():
    """Получить информацию о процессах GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '-q', '-d', 'PIDS'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("\n🔍 GPU Processes:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Process ID' in line or 'Used GPU Memory' in line or 'Process Name' in line:
                    print(f"   {line.strip()}")
    except Exception as e:
        print(f"❌ Не удалось получить информацию о процессах: {e}")

def stress_test_with_monitoring():
    """Стресс-тест с мониторингом"""
    print("🚀 Запуск GPU Stress Test с мониторингом")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA недоступна!")
        return
    
    # Информация о GPU
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_memory:.1f} GB")
    
    # Запускаем мониторинг
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    try:
        print("\n🔥 Запуск интенсивных вычислений...")
        print("💡 Смотрите статистику выше, она обновляется каждые 2 секунды")
        
        # Заполняем память
        print("\n📊 Заполняем VRAM...")
        tensors = []
        for i in range(15):
            try:
                tensor = torch.randn(2048, 2048, 64, device='cuda', dtype=torch.float32)
                tensors.append(tensor)
                print(f"   Тензор {i+1}: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            except torch.cuda.OutOfMemoryError:
                print(f"   Достигнут лимит памяти на тензоре {i+1}")
                break
        
        # Интенсивные вычисления
        print("\n🔥 Интенсивные матричные вычисления...")
        size = 6144
        A = torch.randn(size, size, device='cuda', requires_grad=True)
        B = torch.randn(size, size, device='cuda', requires_grad=True)
        
        for iteration in range(100):
            # Матричное умножение
            C = torch.matmul(A, B)
            
            # Дополнительные операции
            D = torch.sin(C) + torch.cos(C)
            E = torch.tanh(D)
            F = torch.relu(E)
            
            # Backward pass
            loss = torch.sum(F**2)
            loss.backward()
            
            # Обнуляем градиенты
            A.grad = None
            B.grad = None
            
            if iteration % 10 == 0:
                print(f"   Итерация {iteration}/100")
        
        print("\n✅ Стресс-тест завершен!")
        
        # Показываем процессы
        get_process_info()
        
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
    finally:
        monitor.stop_monitoring()
        torch.cuda.empty_cache()
        print("\n🧹 Память GPU очищена")

def simple_nvidia_smi_check():
    """Проверка доступных команд nvidia-smi"""
    print("🔍 Проверка доступных команд nvidia-smi для RTX 4070:")
    print("=" * 60)
    
    commands = [
        ('nvidia-smi', 'Базовая информация'),
        ('nvidia-smi -l 1', 'Мониторинг каждую секунду'),
        ('nvidia-smi -q', 'Детальная информация'),
        ('nvidia-smi -q -d PIDS', 'Информация о процессах'),
        ('nvidia-smi dmon -i 0', 'Альтернативный мониторинг'),
        ('nvidia-smi pmon -i 0', 'Мониторинг процессов (НЕ РАБОТАЕТ на GeForce)')
    ]
    
    for cmd, desc in commands:
        print(f"\n📋 {desc}:")
        print(f"   💻 Команда: {cmd}")
        
        if 'pmon' in cmd:
            print("   ❌ Не поддерживается на GeForce RTX 4070")
        else:
            print("   ✅ Поддерживается")

if __name__ == "__main__":
    print("🎯 GPU Monitoring Test для RTX 4070")
    print("=" * 60)
    
    choice = input("""
Выберите опцию:
1. Запустить стресс-тест с мониторингом
2. Показать доступные команды nvidia-smi
3. Только проверить GPU

Ввод (1-3): """).strip()
    
    if choice == '1':
        stress_test_with_monitoring()
    elif choice == '2':
        simple_nvidia_smi_check()
    elif choice == '3':
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        else:
            print("❌ CUDA недоступна")
    else:
        print("❌ Неверный выбор")