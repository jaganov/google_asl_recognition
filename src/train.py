# train.py
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

class ASLTrainer:
    """Тренер для ASL модели (как у победителя) с оптимизациями для RTX 4070"""
    
    def __init__(self, 
                 data_dir: str = "../data/google_asl_signs",
                 model_dir: str = "models",
                 max_len: int = 384,
                 batch_size: int = 12,  # Оптимально для RTX 4070 12GB VRAM
                 dim: int = 192,
                 lr: float = 5e-4,
                 epochs: int = 400,
                 device: str = None,
                 use_augmentations: bool = True,
                 use_mixed_precision: bool = True,  # Mixed precision для RTX 4070
                 gradient_clip_val: float = 1.0,   # Gradient clipping
                 gradient_accumulation_steps: int = 4):  # Эффективный batch size = 48
        
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
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # CUDA оптимизации для RTX 4070
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Оптимизация памяти
            torch.cuda.empty_cache()
        
        print(f"🎯 Тренер настроен (оптимизирован для RTX 4070):")
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
        
        # Инициализация компонентов
        self._setup_components()
        
    def _setup_components(self):
        """Настройка компонентов"""
        # Загрузчик данных с аугментациями (оптимизирован для памяти)
        self.dataloader = ASLDataLoader(
            data_dir=str(self.data_dir),
            batch_size=self.batch_size,
            max_len=self.max_len,
            preprocessor=None,  # DataLoader создаст препроцессор сам
            num_workers=4  # Оптимально для RTX 4070
        )
        
        # Используем препроцессор из DataLoader
        self.preprocessor = self.dataloader.preprocessor
        
        # Получаем DataLoader'ы с аугментациями для тренировки
        self.train_loader, self.val_loader, self.test_loader = self.dataloader.get_dataloaders(
            augment_train=self.use_augmentations
        )
        
        # Вычисляем размерность входных фич из первого батча
        sample_batch = next(iter(self.train_loader))
        input_dim = sample_batch['features'].shape[-1]  # Размерность features
        
        # Модель
        self.model = get_model(
            input_dim=input_dim,
            num_classes=self.dataloader.num_classes,
            max_len=self.max_len,
            dim=self.dim
        ).to(self.device)
        
        # PyTorch 2.0+ compile для ускорения (отключен для Windows)
        import platform
        if hasattr(torch, 'compile') and self.device.type == 'cuda' and platform.system() != 'Windows':
            try:
                # Пробуем более безопасный режим для Linux/Mac
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("   ✅ PyTorch 2.0+ compile включен (reduce-overhead mode)")
            except Exception as e:
                print(f"   ⚠️ PyTorch compile недоступен: {e}")
                print("   ℹ️ Продолжаем без compile")
                # Убираем compile если он не работает
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
        else:
            if platform.system() == 'Windows':
                print("   ℹ️ PyTorch compile отключен для Windows (избегаем проблем с Triton)")
            else:
                print("   ℹ️ PyTorch compile недоступен (продолжаем без него)")
        
        # Loss function (как у победителя)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer (как у победителя: AdamW)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler (как у победителя: CosineDecay)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        
        # Веса классов для сбалансированного обучения
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
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Перемещаем данные на device
            features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if self.use_mixed_precision and self.device.type == 'cuda':
                try:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(features)
                        loss = self.criterion(outputs, labels)
                        # Нормализуем loss для gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                except Exception as e:
                    print(f"   ⚠️ Mixed precision недоступен: {e}")
                    print("   ℹ️ Продолжаем без mixed precision")
                    self.use_mixed_precision = False
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
            
            # Mixed precision backward pass
            if self.use_mixed_precision and hasattr(self, 'scaler'):
                try:
                    self.scaler.scale(loss).backward()
                except Exception as e:
                    print(f"   ⚠️ Mixed precision backward недоступен: {e}")
                    print("   ℹ️ Продолжаем без mixed precision")
                    self.use_mixed_precision = False
                    loss.backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_mixed_precision and hasattr(self, 'scaler'):
                    try:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    except Exception as e:
                        print(f"   ⚠️ Mixed precision optimizer недоступен: {e}")
                        print("   ℹ️ Продолжаем без mixed precision")
                        self.use_mixed_precision = False
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                        self.optimizer.step()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Статистика
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Обновляем progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Очистка памяти каждые 50 батчей
            if batch_idx % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Обновляем scheduler
        self.scheduler.step()
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, epoch: int):
        """Валидация с оптимизациями"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Перемещаем данные на device
                features = batch['features'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass (с mixed precision если включен)
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
        print(f"🚀 Начинаем обучение (оптимизировано для RTX 4070)...")
        print(f"   Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        best_val_acc = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(self.epochs):
            # Обучение
            train_loss, train_acc = self.train_epoch(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Валидация
            val_loss, val_acc = self.validate(epoch)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Логирование
            print(f"Epoch {epoch+1}/{self.epochs}:")
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
            
            # Сохраняем каждые 50 эпох или если это лучший результат
            if (epoch + 1) % 50 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
            
            # Ранняя остановка (если нет улучшений 50 эпох)
            if epoch > 50 and max(val_accs[-50:]) < best_val_acc:
                print(f"⏹️ Ранняя остановка на эпохе {epoch+1}")
                break
        
        # Сохраняем финальную модель
        self.save_checkpoint(self.epochs-1, val_acc)
        
        # Сохраняем историю обучения
        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
        
        history_path = self.model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✅ Обучение завершено!")
        print(f"   Лучшая валидационная точность: {best_val_acc:.2f}%")
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
    """Основная функция с оптимизациями для RTX 4070"""
    print("🤟 ОБУЧЕНИЕ ASL МОДЕЛИ (оптимизировано для RTX 4070)")
    print("=" * 70)
    
    # Создаем тренер с оптимизациями для RTX 4070
    trainer = ASLTrainer(
        data_dir="../data/google_asl_signs",
        model_dir="models",
        max_len=384,
        batch_size=12,  # Оптимально для RTX 4070 12GB VRAM
        dim=192,  # Можно использовать 384 для большей модели
        lr=5e-4,
        epochs=400,
        use_augmentations=True,  # Включаем аугментации как у победителя
        use_mixed_precision=True,  # Mixed precision для ускорения
        gradient_clip_val=1.0,  # Gradient clipping для стабильности
        gradient_accumulation_steps=4  # Эффективный batch size = 48
    )
    
    # Запускаем обучение
    history = trainer.train()
    
    print("🎉 Обучение завершено успешно!")

if __name__ == "__main__":
    main() 