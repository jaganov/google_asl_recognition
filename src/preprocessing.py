# preprocessing.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from pathlib import Path

class ASLPreprocessor(nn.Module):
    """PyTorch препроцессор для Google ASL Signs dataset"""
    
    def __init__(self, 
                 max_len: int = 384,
                 point_landmarks: Optional[List[int]] = None):
        super().__init__()
        self.max_len = max_len
        
        # Важные landmark точки (адаптировано под реальные данные)
        if point_landmarks is None:
            # Используем все доступные точки (как у победителя, но адаптировано)
            # В реальных данных у нас меньше точек, поэтому берем все
            self.point_landmarks = None  # Будем использовать все точки
        else:
            self.point_landmarks = point_landmarks
            
        # Количество точек будет определено при загрузке первого файла
        self.total_landmarks = None
        
        # Кэш для загруженных файлов (значительно ускоряет загрузку)
        self._file_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 1000  # Максимальный размер кэша
        
        print(f"🎯 Препроцессор настроен (адаптировано под реальные данные):")
        print(f"   Максимальная длина последовательности: {max_len}")
        print(f"   Используемые точки: все доступные")
        print(f"   Кэширование файлов: включено (макс. {self._max_cache_size} файлов)")
    

    
    def load_landmark_file(self, file_path: str) -> torch.Tensor:
        """
        Загружает landmark файл и преобразует в тензор (с кэшированием)
        Input: путь к parquet файлу
        Output: (frames, landmarks, 3) - координаты x, y, z
        """
        # Проверяем кэш
        if file_path in self._file_cache:
            self._cache_hits += 1
            return self._file_cache[file_path]
        
        self._cache_misses += 1
        
        # Загружаем parquet файл
        df = pd.read_parquet(file_path)
        
        # Получаем уникальные кадры и сортируем их
        frames = sorted(df['frame'].unique())
        
        # Получаем все уникальные landmark_index
        all_landmarks = sorted(df['landmark_index'].unique())
        
        # Если это первый файл, устанавливаем total_landmarks
        if self.total_landmarks is None:
            self.total_landmarks = len(all_landmarks)
            # Используем глобальную переменную для отслеживания
            if not hasattr(ASLPreprocessor, '_landmarks_printed'):
                print(f"   Определено количество landmark точек: {self.total_landmarks}")
                print(f"   (Это сообщение выводится только один раз)")
                ASLPreprocessor._landmarks_printed = True
            else:
                # Если флаг уже установлен, просто устанавливаем total_landmarks без вывода
                pass
        
        # Создаем тензор для хранения всех landmarks
        # Размер: (frames, total_landmarks, 3)
        landmarks_tensor = torch.full((len(frames), self.total_landmarks, 3), 
                                    float('nan'), dtype=torch.float32)
        
        # Заполняем тензор данными
        for frame_idx, frame in enumerate(frames):
            # Данные для текущего кадра
            frame_data = df[df['frame'] == frame]
            
            for landmark_idx, point_idx in enumerate(all_landmarks):
                # Ищем точку с нужным индексом
                point_data = frame_data[frame_data['landmark_index'] == point_idx]
                
                if len(point_data) > 0:
                    # Берем первую найденную точку
                    coords = point_data[['x', 'y', 'z']].iloc[0].values
                    landmarks_tensor[frame_idx, landmark_idx] = torch.tensor(coords)
        
        # Сохраняем в кэш (с ограничением размера)
        if len(self._file_cache) < self._max_cache_size:
            self._file_cache[file_path] = landmarks_tensor
        else:
            # Если кэш полный, удаляем самый старый элемент
            oldest_key = next(iter(self._file_cache))
            del self._file_cache[oldest_key]
            self._file_cache[file_path] = landmarks_tensor
        
        return landmarks_tensor
    
    def forward(self, x):
        """
        Препроцессинг landmarks (как у победителя)
        Input: (batch, frames, landmarks, channels) 
        Output: (batch, frames, features)
        """
        batch_size, frames, landmarks, channels = x.shape
        
        # 1. Нормализация по центру кадра (адаптировано под реальные данные)
        # Используем среднее по всем точкам как референс
        nose_mean = torch.mean(x[:, :, :, :2], dim=(1,2), keepdim=True)
        nose_mean = torch.where(torch.isnan(nose_mean), 
                              torch.tensor(0.5, device=x.device, dtype=x.dtype), 
                              nose_mean)
        
        # 2. Используем все landmark точки (адаптировано под реальные данные)
        x_selected = x
        
        # 3. Стандартизация (как у победителя)
        x_coords = x_selected[:, :, :, :2]  # Только x, y координаты
        std = torch.std(x_coords, dim=(1,2), keepdim=True, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        
        x_norm = (x_coords - nose_mean) / std
        
        # 4. Обрезка по времени
        if frames > self.max_len:
            x_norm = x_norm[:, :self.max_len]
            frames = self.max_len
        
        # 5. Motion features (точно как у победителя)
        # dx = x[t+1] - x[t] (lag1)
        if frames > 1:
            dx = torch.zeros_like(x_norm)
            dx[:, :-1] = x_norm[:, 1:] - x_norm[:, :-1]
        else:
            dx = torch.zeros_like(x_norm)
        
        # dx2 = x[t+2] - x[t] (lag2)
        if frames > 2:
            dx2 = torch.zeros_like(x_norm)
            dx2[:, :-2] = x_norm[:, 2:] - x_norm[:, :-2]
        else:
            dx2 = torch.zeros_like(x_norm)
        
        # 6. Объединяем все фичи (как у победителя)
        # Flatten landmarks dimension: (batch, frames, landmarks*2)
        x_flat = x_norm.view(batch_size, frames, -1)
        dx_flat = dx.view(batch_size, frames, -1) 
        dx2_flat = dx2.view(batch_size, frames, -1)
        
        # Concatenate: position + velocity + acceleration
        features = torch.cat([x_flat, dx_flat, dx2_flat], dim=-1)
        
        # 7. Заменяем NaN на 0 (как у победителя)
        features = torch.where(torch.isnan(features), 
                             torch.zeros_like(features), 
                             features)
        
        return features
    
    def get_cache_stats(self) -> dict:
        """Возвращает статистику кэша"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._file_cache),
            'max_cache_size': self._max_cache_size
        }
    
    def clear_cache(self):
        """Очищает кэш"""
        self._file_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        print("🧹 Кэш препроцессора очищен")

class ASLDataLoader:
    """Загрузчик данных для Google ASL Signs"""
    
    def __init__(self, 
                 data_dir: str = "../data/google_asl_signs",
                 max_len: int = 384,
                 batch_size: int = 32,
                 preprocessor: Optional[ASLPreprocessor] = None):
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.batch_size = batch_size
        
        if preprocessor is None:
            self.preprocessor = ASLPreprocessor(max_len=max_len)
        else:
            self.preprocessor = preprocessor
            
        # Загружаем метаданные
        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        self.sign_mapping = self._load_sign_mapping()
        
        print(f"📊 Загружено {len(self.train_df)} записей")
        print(f"🎯 Уникальных знаков: {self.train_df['sign'].nunique()}")
    
    def _load_sign_mapping(self) -> Dict[str, int]:
        """Загружает маппинг знаков к индексам"""
        with open(self.data_dir / "sign_to_prediction_index_map.json", 'r') as f:
            import json
            return json.load(f)
    
    def load_sequence(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Загружает одну последовательность landmarks
        Returns: (landmarks_tensor, label)
        """
        # Полный путь к файлу
        full_path = self.data_dir / file_path
        
        # Загружаем landmarks
        landmarks = self.preprocessor.load_landmark_file(str(full_path))
        
        # Получаем метку
        row = self.train_df[self.train_df['path'] == file_path].iloc[0]
        label = self.sign_mapping[row['sign']]
        
        return landmarks, label
    
    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Загружает батч данных
        Returns: (batch_landmarks, batch_labels)
        """
        batch_landmarks = []
        batch_labels = []
        
        for idx in indices:
            file_path = self.train_df.iloc[idx]['path']
            landmarks, label = self.load_sequence(file_path)
            batch_landmarks.append(landmarks)
            batch_labels.append(label)
        
        # Паддинг до максимальной длины в батче
        max_frames = max(landmarks.shape[0] for landmarks in batch_landmarks)
        max_frames = min(max_frames, self.max_len)
        
        # Создаем тензоры
        batch_tensor = torch.zeros(len(batch_landmarks), max_frames, 
                                 self.preprocessor.total_landmarks, 3)
        
        for i, landmarks in enumerate(batch_landmarks):
            frames = min(landmarks.shape[0], max_frames)
            batch_tensor[i, :frames] = landmarks[:frames]
        
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        
        return batch_tensor, batch_labels

def test_preprocessor():
    """Тест препроцессора с реальными данными"""
    print("🧪 Тестируем препроцессор с реальными данными...")
    
    # Создаем препроцессор
    preprocessor = ASLPreprocessor(max_len=64)
    
    # Создаем загрузчик данных
    data_loader = ASLDataLoader(preprocessor=preprocessor)
    
    # Тестируем на одной последовательности
    test_file = data_loader.train_df.iloc[0]['path']
    print(f"📁 Тестовый файл: {test_file}")
    
    try:
        landmarks, label = data_loader.load_sequence(test_file)
        print(f"   Загружено landmarks: {landmarks.shape}")
        print(f"   Метка: {label}")
        
        # Препроцессинг
        with torch.no_grad():
            # Добавляем batch dimension
            landmarks_batch = landmarks.unsqueeze(0)
            features = preprocessor(landmarks_batch)
        
        print(f"   Input shape: {landmarks_batch.shape}")
        print(f"   Output shape: {features.shape}")
        print(f"   NaN в output: {torch.isnan(features).sum().item()}")
        print(f"   Output range: [{features.min():.3f}, {features.max():.3f}]")
        
        return features.shape[-1]  # Количество фич
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return None

if __name__ == "__main__":
    feature_dim = test_preprocessor()
    if feature_dim:
        print(f"✅ Препроцессор готов! Feature dimension: {feature_dim}")
    else:
        print("❌ Ошибка в препроцессоре")