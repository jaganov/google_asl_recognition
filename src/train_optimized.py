# train_optimized.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm
import time
import os
from pathlib import Path
import json
from datetime import datetime
import platform
import psutil

# Подавляем ошибки Triton на Windows
if platform.system() == 'Windows':
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        print("🔧 Подавлены ошибки Triton для Windows")
    except:
        pass

# Настройка TensorFloat32 для лучшей производительности
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    print("🔧 Включен TensorFloat32 для лучшей производительности")

from data_loader import ASLDataLoader
from preprocessing import ASLPreprocessor
from models import get_model, ASLEnsemble

class OptimizedASLTrainer:
    """Оптимизированный тренер для ASL модели с исправлениями производительности"""
    
    def __init__(self, 
                 data_dir: str = "../data/google_asl_signs",
                 model_dir: str = "models",
                 max_len: int = 384,
                 batch_size: int = 16,  # Увеличили для RTX 4070
                 dim: int = 192,
                 lr: float = 5e-4,
                 epochs: int = 400,
                 device: str = None,
                 use_augmentations: bool = True,
                 use_mixed_precision: bool = True,
                 gradient_clip_val: float = 1.0,
                 gradient_accumulation_steps: int = 3,  # Уменьшили для ускорения
                 num_workers: int = 2,  # Уменьшили для Windows
                 pin_memory: bool = True):
        
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.max_len = max_len
        self.batch_size = batch_size
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.use_augmentations = use_augmentations
        self.use_mixed_precision = use_mixed_precision
        self.gradient_clip_val = gradient_clip_val
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # CUDA оптимизации
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
        
        print(f"🎯 Оптимизированный тренер настроен:")
        print(f"   Device: {self.device}")
        print(f"   Максимальная длина: {max_len}")
        print(f"   Размер батча: {batch_size}")
        print(f"   Эффективный batch size: {batch_size * gradient_accumulation_steps}")
        print(f"   Размерность модели: {dim}")
        print(f"   Learning rate: {lr}")
        print(f"   Эпохи: {epochs}")
        print(f"   Аугментации: {'Включены' if use_augmentations else 'Отключены'}")
        print(f"   Mixed Precision: {'Включен' if use_mixed_precision else 'Отключен'}")
        print(f"   Gradient Clipping: {gradient_clip_val}")
        print(f"   Gradient Accumulation: {gradient_accumulation_steps} steps")
        print(f"   Num Workers: {num_workers}")
        print(f"   Pin Memory: {pin_memory}")
        
        # Инициализация компонентов
        self._setup_components()
        
    def _setup_components(self):
        """Настройка компонентов с оптимизациями"""
        print("\n📦 Настройка компонентов...")
        
        # Создаем препроцессор с увеличенным кэшем
        self.preprocessor = ASLPreprocessor(max_len=self.max_len)
        self.preprocessor._max_cache_size = 2000  # Увеличиваем кэш
        
        # Загрузчик данных с оптимизациями
        self.dataloader = ASLDataLoader(
            data_dir=str(self.data_dir),
            batch_size=self.batch_size,
            max_len=self.max_len,
            preprocessor=self.preprocessor,
            num_workers=self.num_workers
        )
        
        # Получаем DataLoader'ы
        print("📂 Создаем DataLoader'ы...")
        self.train_loader, self.val_loader, self.test_loader = self.dataloader.get_dataloaders(
            augment_train=self.use_augmentations
        )
        
        # Вычисляем размерность входных фич из первого батча
        print("🔍 Определяем размерность входных данных...")
        sample_batch = next(iter(self.train_loader))
        input_dim = sample_batch['features'].shape[-1]
        
        # Модель
        print("🤖 Создаем модель...")
        self.model = get_model(
            input_dim=input_dim,
            num_classes=self.dataloader.num_classes,
            max_len=self.max_len,
            dim=self.dim
        ).to(self.device)
        
        # PyTorch 2.0+ compile (только для Linux/Mac)
        if hasattr(torch, 'compile') and self.device.type == 'cuda' and platform.system() != 'Windows':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("   ✅ PyTorch 2.0+ compile включен")
            except Exception as e:
                print(f"   ⚠️ PyTorch compile недоступен: {e}")
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
        else:
            print("   ℹ️ PyTorch compile отключен (Windows или недоступен)")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        
        # Веса классов
        self.class_weights = self.dataloader.get_class_weights('train').to(self.device)
        
        print(f"✅ Компоненты настроены:")
        print(f"   Размерность входных фич: {input_dim}")
        print(f"   Количество классов: {self.dataloader.num_classes}")
        print(f"   Параметры модели: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Статистика кэша препроцессора
        cache_stats = self.preprocessor.get_cache_stats()
        print(f"   Кэш препроцессора: {cache_stats['cache_size']}/{cache_stats['max_cache_size']} файлов")
        
        # Проверка памяти GPU
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU память: {gpu_memory:.1f} GB")
    
    def train_epoch(self, epoch: int):
        """Обучение одной эпохи с оптимизациями"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Измеряем время эпохи
        epoch_start_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Перемещаем данные на device
            features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if self.use_mixed_precision and self.device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
            
            # Mixed precision backward pass
            if self.use_mixed_precision and hasattr(self, 'scaler'):
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_mixed_precision and hasattr(self, 'scaler'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Статистика
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Обновляем progress bar с дополнительной информацией
            epoch_time = time.time() - epoch_start_time
            batches_per_sec = (batch_idx + 1) / epoch_time if epoch_time > 0 else 0
            
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}',
                'Speed': f'{batches_per_sec:.1f} batch/s'
            })
            
            # Очистка памяти каждые 20 батчей (уменьшили частоту)
            if batch_idx % 20 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Обновляем scheduler
        self.scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f"   ⏱️ Время эпохи: {epoch_time:.1f} сек")
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, epoch: int):
        """Валидация с оптимизациями"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Перемещаем данные на device
                features = batch['features'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast('cuda'):
                        outputs = self.model(features)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                
                # Статистика
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_time = time.time() - val_start_time
        print(f"   ⏱️ Время валидации: {val_time:.1f} сек")
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': {
                'max_len': self.max_len,
                'batch_size': self.batch_size,
                'dim': self.dim,
                'lr': self.lr,
                'use_augmentations': self.use_augmentations,
                'use_mixed_precision': self.use_mixed_precision,
                'gradient_clip_val': self.gradient_clip_val,
                'gradient_accumulation_steps': self.gradient_accumulation_steps
            }
        }
        
        # Сохраняем scaler для mixed precision
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Сохраняем последний чекпоинт
        checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Сохраняем лучший чекпоинт
        if is_best:
            best_path = self.model_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Сохранен лучший чекпоинт: {best_path}")
    
    def train(self):
        """Основной цикл обучения с оптимизациями"""
        print(f"🚀 Начинаем оптимизированное обучение...")
        print(f"   Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        best_val_acc = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        total_start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Обучение
            train_loss, train_acc = self.train_epoch(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Валидация
            val_loss, val_acc = self.validate(epoch)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            # Логирование
            print(f"Epoch {epoch+1}/{self.epochs} ({epoch_time:.1f} сек):")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Информация о памяти GPU
            if self.device.type == 'cuda':
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"  GPU Memory: {gpu_memory_used:.1f}GB used, {gpu_memory_cached:.1f}GB cached")
            
            # Сохранение чекпоинта
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                print(f"🎉 Новый лучший результат: {val_acc:.2f}%")
            
            # Сохраняем каждые 25 эпох или если это лучший результат
            if (epoch + 1) % 25 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
            
            # Ранняя остановка (если нет улучшений 30 эпох)
            if epoch > 30 and max(val_accs[-30:]) < best_val_acc:
                print(f"⏹️ Ранняя остановка на эпохе {epoch+1}")
                break
        
        total_time = time.time() - total_start_time
        
        # Сохраняем финальную модель
        self.save_checkpoint(self.epochs-1, val_acc)
        
        # Сохраняем историю обучения
        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'total_training_time': total_time
        }
        
        history_path = self.model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✅ Обучение завершено!")
        print(f"   Лучшая валидационная точность: {best_val_acc:.2f}%")
        print(f"   Общее время обучения: {total_time/3600:.1f} часов")
        print(f"   Время окончания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Статистика кэша препроцессора
        cache_stats = self.preprocessor.get_cache_stats()
        print(f"📊 Статистика кэша препроцессора:")
        print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"   Cache hits: {cache_stats['cache_hits']:,}")
        print(f"   Cache misses: {cache_stats['cache_misses']:,}")
        print(f"   Файлов в кэше: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        
        return history

def main():
    """Основная функция с оптимизациями"""
    print("🤟 ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ ASL МОДЕЛИ")
    print("=" * 70)
    
    # Проверяем системные ресурсы
    print("🔍 Проверка системных ресурсов:")
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    print(f"   CPU: {cpu_count} ядер")
    print(f"   RAM: {memory.total / 1024**3:.1f} GB")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
    
    # Создаем оптимизированный тренер
    trainer = OptimizedASLTrainer(
        data_dir="../data/google_asl_signs",
        model_dir="models",
        max_len=384,
        batch_size=16,  # Увеличили для RTX 4070
        dim=192,
        lr=5e-4,
        epochs=400,
        use_augmentations=True,
        use_mixed_precision=True,
        gradient_clip_val=1.0,
        gradient_accumulation_steps=3,  # Уменьшили для ускорения
        num_workers=2,  # Уменьшили для Windows
        pin_memory=True
    )
    
    # Запускаем обучение
    history = trainer.train()
    
    print("🎉 Оптимизированное обучение завершено успешно!")

if __name__ == "__main__":
    main() 