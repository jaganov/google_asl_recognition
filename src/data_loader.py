# data_loader.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from preprocessing import ASLPreprocessor
from augmentations import ASLAugmentations

class ASLDataset(Dataset):
    """Dataset для Google ASL Signs"""
    
    def __init__(self, 
                 data_dir: str,
                 split_file: str,
                 preprocessor: ASLPreprocessor,
                 max_len: int = 384,
                 augment: bool = False):
        """
        Args:
            data_dir: Путь к директории с данными
            split_file: Файл со сплитом (train.csv, val.csv, test.csv)
            preprocessor: Препроцессор landmarks
            max_len: Максимальная длина последовательности
            augment: Использовать ли аугментацию
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.augment = augment
        
        # Инициализируем аугментации если нужно
        if self.augment:
            self.augmenter = ASLAugmentations()
        
        # Загружаем данные сплита
        self.df = pd.read_csv(self.data_dir / split_file)
        
        # Загружаем маппинг знаков
        with open(self.data_dir / "sign_to_prediction_index_map.json", 'r') as f:
            self.sign_mapping = json.load(f)
        
        print(f"📊 Загружен {split_file}: {len(self.df)} записей")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Загружает одну последовательность"""
        row = self.df.iloc[idx]
        file_path = row['path']
        sign = row['sign']
        
        # Получаем метку
        label = self.sign_mapping[sign]
        
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

class ASLDataLoader:
    """Улучшенный загрузчик данных для Google ASL Signs"""
    
    def __init__(self, 
                 data_dir: str = "../data/google_asl_signs",
                 batch_size: int = 32,
                 max_len: int = 384,
                 num_workers: int = 4,
                 preprocessor: Optional[ASLPreprocessor] = None):
        """
        Args:
            data_dir: Путь к директории с данными
            batch_size: Размер батча
            max_len: Максимальная длина последовательности
            num_workers: Количество воркеров для загрузки
            preprocessor: Препроцессор (если None, создается новый)
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        
        if preprocessor is None:
            self.preprocessor = ASLPreprocessor(max_len=max_len)
        else:
            self.preprocessor = preprocessor
        
        # Загружаем маппинг знаков
        with open(self.data_dir / "sign_to_prediction_index_map.json", 'r') as f:
            self.sign_mapping = json.load(f)
        
        self.num_classes = len(self.sign_mapping)
        print(f"🎯 Настроен загрузчик данных:")
        print(f"   Количество классов: {self.num_classes}")
        print(f"   Размер батча: {batch_size}")
        print(f"   Максимальная длина: {max_len}")
    
    def get_dataloaders(self, 
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1,
                       augment_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Создает DataLoader'ы для train/val/test
        
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
        
        # Создаем датасеты
        train_dataset = ASLDataset(
            data_dir=self.data_dir,
            split_file="splits/train.csv",
            preprocessor=self.preprocessor,
            max_len=self.max_len,
            augment=augment_train
        )
        
        val_dataset = ASLDataset(
            data_dir=self.data_dir,
            split_file="splits/val.csv",
            preprocessor=self.preprocessor,
            max_len=self.max_len,
            augment=False
        )
        
        test_dataset = ASLDataset(
            data_dir=self.data_dir,
            split_file="splits/test.csv",
            preprocessor=self.preprocessor,
            max_len=self.max_len,
            augment=False
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
        
        print(f"📊 Созданы DataLoader'ы:")
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
        
        sign_counts = df['sign'].value_counts()
        
        # Вычисляем веса (обратно пропорционально частоте)
        total_samples = len(df)
        class_weights = total_samples / (len(sign_counts) * sign_counts)
        
        # Сортируем по индексам классов
        class_weights = class_weights.sort_index()
        
        return torch.tensor(class_weights.values, dtype=torch.float32)
    
    def get_sample_batch(self, split: str = 'train') -> Dict:
        """Получает пример батча для тестирования"""
        split_file = f"splits/{split}.csv"
        dataset = ASLDataset(
            data_dir=self.data_dir,
            split_file=split_file,
            preprocessor=self.preprocessor,
            max_len=self.max_len,
            augment=False
        )
        
        # Берем первые несколько образцов
        sample_batch = [dataset[i] for i in range(min(4, len(dataset)))]
        
        return self._collate_fn(sample_batch)

def test_dataloader():
    """Тест загрузчика данных"""
    print("🧪 Тестируем загрузчик данных...")
    
    try:
        # Создаем загрузчик
        dataloader = ASLDataLoader(batch_size=4, max_len=64)
        
        # Получаем сплиты
        train_loader, val_loader, test_loader = dataloader.get_dataloaders()
        
        # Тестируем один батч
        sample_batch = next(iter(train_loader))
        
        print(f"✅ Тест успешен!")
        print(f"   Размер батча: {sample_batch['features'].shape}")
        print(f"   Метки: {sample_batch['labels']}")
        print(f"   Длины: {sample_batch['lengths']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в загрузчике данных: {e}")
        return False

if __name__ == "__main__":
    test_dataloader() 