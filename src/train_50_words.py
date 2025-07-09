# train_50_words.py
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
import pandas as pd
from typing import Optional, Tuple
from torch.utils.data import DataLoader

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

class ASLDataset50Words:
    """Dataset для Google ASL Signs с ограничением до 50 слов"""
    
    def __init__(self, 
                 data_dir: str,
                 split_file: str,
                 preprocessor: ASLPreprocessor,
                 max_len: int = 384,
                 augment: bool = False,
                 num_words: int = 50):
        """
        Args:
            data_dir: Путь к директории с данными
            split_file: Файл со сплитом (train.csv, val.csv, test.csv)
            preprocessor: Препроцессор landmarks
            max_len: Максимальная длина последовательности
            augment: Использовать ли аугментацию
            num_words: Количество слов для обучения (по умолчанию 50)
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.augment = augment
        self.num_words = num_words
        
        # Инициализируем аугментации если нужно
        if self.augment:
            from augmentations import ASLAugmentations
            self.augmenter = ASLAugmentations()
        
        # Загружаем данные сплита
        self.df = pd.read_csv(self.data_dir / split_file)
        
        # Загружаем маппинг знаков
        with open(self.data_dir / "sign_to_prediction_index_map.json", 'r') as f:
            self.sign_mapping = json.load(f)
        
        # Фильтруем только первые num_words знаков
        all_signs = list(self.sign_mapping.keys())
        selected_signs = all_signs[:num_words]
        
        # Создаем новый маппинг только для выбранных знаков
        self.filtered_sign_mapping = {sign: idx for idx, sign in enumerate(selected_signs)}
        
        # Фильтруем данные только для выбранных знаков
        self.df = self.df[self.df['sign'].isin(selected_signs)].copy()
        
        # Обновляем метки в соответствии с новым маппингом
        self.df['label'] = self.df['sign'].map(self.filtered_sign_mapping)
        
        print(f"📊 Загружен {split_file} (фильтровано до {num_words} слов): {len(self.df)} записей")
        print(f"🎯 Выбранные знаки: {', '.join(selected_signs[:10])}{'...' if len(selected_signs) > 10 else ''}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Загружает одну последовательность"""
        row = self.df.iloc[idx]
        file_path = row['path']
        sign = row['sign']
        label = row['label']  # Уже обновленная метка
        
        # Загружаем landmarks
        full_path = self.data_dir / file_path
        landmarks = self.preprocessor.load_landmark_file(str(full_path))
        
        # Обрезка по времени
        if landmarks.shape[0] > self.max_len:
            landmarks = landmarks[:self.max_len]
        
        # Препроцессинг landmarks в features
        landmarks_batch = landmarks.unsqueeze(0)  # (1, frames, landmarks, 3)
        features = self.preprocessor(landmarks_batch).squeeze(0)  # (frames, features)
        
        # Аугментация (если включена) - применяется к features
        if self.augment:
            features = self._augment_features(features)
        
        return {
            'landmarks': features,  # Теперь это features, а не landmarks
            'label': label,
            'sign': sign,
            'file_path': file_path
        }
    
    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """Аугментация features с использованием ASLAugmentations"""
        if not self.augment or not hasattr(self, 'augmenter'):
            return features
            
        # Добавляем batch dimension для аугментаций
        features_batch = features.unsqueeze(0)  # (1, frames, features)
        
        # Применяем аугментации
        augmented = self.augmenter(features_batch)
        
        # Убираем batch dimension
        return augmented.squeeze(0)

class ASLDataLoader50Words:
    """Загрузчик данных для Google ASL Signs с ограничением до 50 слов"""
    
    def __init__(self, 
                 data_dir: str = "../data/google_asl_signs",
                 batch_size: int = 32,
                 max_len: int = 384,
                 num_workers: int = 4,
                 preprocessor: Optional[ASLPreprocessor] = None,
                 num_words: int = 50):
        """
        Args:
            data_dir: Путь к директории с данными
            batch_size: Размер батча
            max_len: Максимальная длина последовательности
            num_workers: Количество воркеров для загрузки
            preprocessor: Препроцессор (если None, создается новый)
            num_words: Количество слов для обучения (по умолчанию 50)
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.num_words = num_words
        
        if preprocessor is None:
            self.preprocessor = ASLPreprocessor(max_len=max_len)
        else:
            self.preprocessor = preprocessor
        
        # Загружаем маппинг знаков для определения количества классов
        with open(self.data_dir / "sign_to_prediction_index_map.json", 'r') as f:
            self.sign_mapping = json.load(f)
        
        # Количество классов теперь равно num_words
        self.num_classes = num_words
        
        print(f"🎯 Настроен загрузчик данных (50 слов):")
        print(f"   Количество классов: {self.num_classes}")
        print(f"   Размер батча: {batch_size}")
        print(f"   Максимальная длина: {max_len}")
    
    def get_dataloaders(self, 
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1,
                       augment_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Создает DataLoader'ы для train/val/test с ограничением до 50 слов
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Проверяем наличие сплитов
        splits_dir = self.data_dir / "splits"
        
        if not splits_dir.exists():
            print("⚠️ Сплиты не найдены. Создаем новые...")
            from data_utils import ASLDataAnalyzer
            analyzer = ASLDataAnalyzer(str(self.data_dir))
            analyzer.create_balanced_splits(train_ratio, val_ratio, test_ratio)
        
        # Создаем датасеты с ограничением до 50 слов
        train_dataset = ASLDataset50Words(
            data_dir=self.data_dir,
            split_file="splits/train.csv",
            preprocessor=self.preprocessor,
            max_len=self.max_len,
            augment=augment_train,
            num_words=self.num_words
        )
        
        val_dataset = ASLDataset50Words(
            data_dir=self.data_dir,
            split_file="splits/val.csv",
            preprocessor=self.preprocessor,
            max_len=self.max_len,
            augment=False,
            num_words=self.num_words
        )
        
        test_dataset = ASLDataset50Words(
            data_dir=self.data_dir,
            split_file="splits/test.csv",
            preprocessor=self.preprocessor,
            max_len=self.max_len,
            augment=False,
            num_words=self.num_words
        )
        
        # Создаем DataLoader'ы
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        print(f"📊 Созданы DataLoader'ы (50 слов):")
        print(f"   Train: {len(train_loader)} батчей")
        print(f"   Val: {len(val_loader)} батчей")
        print(f"   Test: {len(test_loader)} батчей")
        
        return train_loader, val_loader, test_loader
    
    def _collate_fn(self, batch):
        """
        Функция для объединения батча
        Обрабатывает последовательности разной длины
        """
        features_list = [item['landmarks'] for item in batch]  # Теперь это features
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        
        # Находим максимальную длину в батче
        max_frames = max(features.shape[0] for features in features_list)
        max_frames = min(max_frames, self.max_len)
        
        # Получаем размер features из первого элемента
        feature_dim = features_list[0].shape[1]
        
        # Создаем тензор для батча
        batch_size = len(features_list)
        batch_tensor = torch.zeros(batch_size, max_frames, feature_dim)
        
        # Заполняем тензор
        for i, features in enumerate(features_list):
            frames = min(features.shape[0], max_frames)
            batch_tensor[i, :frames] = features[:frames]
        
        return {
            'features': batch_tensor,  # Переименовываем в features
            'labels': labels,
            'lengths': torch.tensor([min(features.shape[0], max_frames) 
                                   for features in features_list])
        }
    
    def get_class_weights(self, split: str = 'train') -> torch.Tensor:
        """Вычисляет веса классов для сбалансированного обучения"""
        split_file = f"splits/{split}.csv"
        df = pd.read_csv(self.data_dir / split_file)
        
        # Загружаем маппинг знаков
        with open(self.data_dir / "sign_to_prediction_index_map.json", 'r') as f:
            sign_mapping = json.load(f)
        
        # Фильтруем только первые num_words знаков
        all_signs = list(sign_mapping.keys())
        selected_signs = all_signs[:self.num_words]
        
        # Фильтруем данные только для выбранных знаков
        df = df[df['sign'].isin(selected_signs)].copy()
        
        sign_counts = df['sign'].value_counts()
        
        # Вычисляем веса (обратно пропорционально частоте)
        total_samples = len(df)
        class_weights = total_samples / (len(sign_counts) * sign_counts)
        
        # Сортируем по индексам классов
        class_weights = class_weights.sort_index()
        
        return torch.tensor(class_weights.values, dtype=torch.float32)

class OptimizedASLTrainer50Words:
    """Оптимизированный тренер для ASL модели с ограничением до 50 слов"""
    
    def __init__(self, 
                 data_dir: str = "../data/google_asl_signs",
                 model_dir: str = "models",
                 max_len: int = 384,
                 batch_size: int = 16,
                 dim: int = 192,
                 lr: float = 5e-4,
                 epochs: int = 100,  # Уменьшили для быстрого тестирования
                 device: str = None,
                 use_augmentations: bool = True,
                 use_mixed_precision: bool = True,
                 gradient_clip_val: float = 1.0,
                 gradient_accumulation_steps: int = 3,
                 num_workers: int = 2,
                 pin_memory: bool = True,
                 num_words: int = 50):
        
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
        self.num_words = num_words
        
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
        
        print(f"🎯 Оптимизированный тренер настроен (50 слов):")
        print(f"   Device: {self.device}")
        print(f"   Количество слов: {num_words}")
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
        
        # Загрузчик данных с ограничением до 50 слов
        self.dataloader = ASLDataLoader50Words(
            data_dir=str(self.data_dir),
            batch_size=self.batch_size,
            max_len=self.max_len,
            preprocessor=self.preprocessor,
            num_workers=self.num_workers,
            num_words=self.num_words
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
            num_classes=self.dataloader.num_classes,  # Теперь это num_words
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
            
            # Очистка памяти каждые 20 батчей
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
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'num_words': self.num_words
            }
        }
        
        # Сохраняем scaler для mixed precision
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Сохраняем последний чекпоинт
        checkpoint_path = self.model_dir / f"checkpoint_50words_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Сохраняем лучший чекпоинт
        if is_best:
            best_path = self.model_dir / "best_model_50words.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Сохранен лучший чекпоинт (50 слов): {best_path}")
    
    def train(self):
        """Основной цикл обучения с оптимизациями"""
        print(f"🚀 Начинаем оптимизированное обучение (50 слов)...")
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
            
            # Сохраняем каждые 10 эпох или если это лучший результат (уменьшили для быстрого тестирования)
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
            
            # Ранняя остановка (если нет улучшений 15 эпох)
            if epoch > 15 and max(val_accs[-15:]) < best_val_acc:
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
            'total_training_time': total_time,
            'num_words': self.num_words
        }
        
        history_path = self.model_dir / "training_history_50words.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✅ Обучение завершено (50 слов)!")
        print(f"   Лучшая валидационная точность: {best_val_acc:.2f}%")
        print(f"   Общее время обучения: {total_time/60:.1f} минут")
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
    """Основная функция с оптимизациями для 50 слов"""
    print("🤟 ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ ASL МОДЕЛИ (50 СЛОВ)")
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
    
    # Создаем оптимизированный тренер для 50 слов
    trainer = OptimizedASLTrainer50Words(
        data_dir="../data/google_asl_signs",
        model_dir="models",
        max_len=384,
        batch_size=16,
        dim=192,
        lr=5e-4,
        epochs=100,  # Уменьшили для быстрого тестирования
        use_augmentations=True,
        use_mixed_precision=True,
        gradient_clip_val=1.0,
        gradient_accumulation_steps=3,
        num_workers=2,
        pin_memory=True,
        num_words=50  # Ограничиваем до 50 слов
    )
    
    # Запускаем обучение
    history = trainer.train()
    
    print("🎉 Оптимизированное обучение (50 слов) завершено успешно!")

if __name__ == "__main__":
    main() 