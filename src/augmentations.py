# augmentations.py
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional, Tuple

class ASLAugmentations:
    """
    Аугментации для Google ASL Signs dataset
    Основано на победном решении: https://www.kaggle.com/competitions/asl-signs/discussion/406684
    """
    
    def __init__(self, 
                 temporal_prob: float = 0.5,
                 spatial_prob: float = 0.7,
                 noise_prob: float = 0.3,
                 mask_prob: float = 0.2):
        """
        Args:
            temporal_prob: Вероятность применения временных аугментаций
            spatial_prob: Вероятность применения пространственных аугментаций
            noise_prob: Вероятность добавления шума
            mask_prob: Вероятность маскирования
        """
        self.temporal_prob = temporal_prob
        self.spatial_prob = spatial_prob
        self.noise_prob = noise_prob
        self.mask_prob = mask_prob
        
        # Определяем важные landmark точки для аугментаций
        # Face landmarks (0-67) - первые 68 точек обычно лицо
        self.face_landmarks = list(range(0, 68))
        # Остальные точки - руки и тело (адаптировано под реальные данные)
        self.body_landmarks = list(range(68, 468))
        
        # Все landmark точки
        self.all_landmarks = list(range(0, 468))  # Реальные данные: 468 точек
        
        print(f"🎯 Аугментации настроены:")
        print(f"   Face landmarks: {len(self.face_landmarks)} точек")
        print(f"   Body landmarks: {len(self.body_landmarks)} точек")
        print(f"   Total landmarks: {len(self.all_landmarks)} точек")
    
    def temporal_resample(self, x: torch.Tensor, min_factor: float = 0.8, max_factor: float = 1.2) -> torch.Tensor:
        """
        Временное изменение скорости (temporal resampling)
        Основано на победном решении
        Сохраняет исходную длину последовательности
        """
        if random.random() > self.temporal_prob:
            return x
            
        batch_size, seq_len, features = x.shape
        
        # Случайный фактор изменения скорости
        factor = random.uniform(min_factor, max_factor)
        
        # Создаем новые индексы для исходной длины
        indices = torch.linspace(0, seq_len - 1, seq_len, device=x.device) * factor
        indices = torch.clamp(indices, 0, seq_len - 1)
        
        indices_floor = torch.floor(indices).long()
        indices_ceil = torch.ceil(indices).long()
        
        # Clamp indices
        indices_floor = torch.clamp(indices_floor, 0, seq_len - 1)
        indices_ceil = torch.clamp(indices_ceil, 0, seq_len - 1)
        
        weight = indices - indices_floor.float()
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        x_floor = x[:, indices_floor]  # (batch, seq_len, features)
        x_ceil = x[:, indices_ceil]
        
        x_resampled = x_floor * (1 - weight) + x_ceil * weight
        
        return x_resampled
    
    def temporal_masking(self, x: torch.Tensor, max_mask_ratio: float = 0.15) -> torch.Tensor:
        """
        Временное маскирование (temporal masking)
        Основано на победном решении
        """
        if random.random() > self.mask_prob:
            return x
            
        batch_size, seq_len, features = x.shape
        mask_len = int(seq_len * random.uniform(0, max_mask_ratio))
        
        if mask_len > 0:
            start_idx = random.randint(0, seq_len - mask_len)
            x[:, start_idx:start_idx + mask_len] = 0
        
        return x
    
    def spatial_noise(self, x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """
        Добавление пространственного шума
        Основано на победном решении
        """
        if random.random() > self.noise_prob:
            return x
            
        noise = torch.randn_like(x) * std
        return x + noise
    
    def spatial_flip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Горизонтальное отражение (меняем знак x координат)
        Основано на победном решении
        """
        if random.random() > self.spatial_prob:
            return x
            
        # Применяем отражение только к x координатам
        # Структура features: [x1, y1, z1, x2, y2, z2, ...]
        x_coords = torch.arange(0, x.shape[-1], 3, device=x.device)
        x[:, :, x_coords] = -x[:, :, x_coords]
        
        return x
    
    def spatial_rotation(self, x: torch.Tensor, max_angle: float = 15.0) -> torch.Tensor:
        """
        Пространственное вращение в плоскости XY
        Основано на победном решении
        """
        if random.random() > self.spatial_prob:
            return x
            
        angle = random.uniform(-max_angle, max_angle) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Применяем вращение к каждой паре (x, y) координат
        for i in range(0, x.shape[-1], 3):  # Каждая тройка (x, y, z)
            if i + 1 < x.shape[-1]:  # Проверяем наличие y координаты
                x_coords = x[:, :, i:i+1]
                y_coords = x[:, :, i+1:i+2]
                
                # Вращение
                new_x = cos_a * x_coords - sin_a * y_coords
                new_y = sin_a * x_coords + cos_a * y_coords
                
                x[:, :, i:i+1] = new_x
                x[:, :, i+1:i+2] = new_y
        
        return x
    
    def spatial_scale(self, x: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """
        Пространственное масштабирование
        Основано на победном решении
        """
        if random.random() > self.spatial_prob:
            return x
            
        scale = random.uniform(*scale_range)
        
        # Применяем масштабирование к x, y координатам
        for i in range(0, x.shape[-1], 3):  # Каждая тройка (x, y, z)
            if i + 1 < x.shape[-1]:
                x[:, :, i:i+2] *= scale
        
        return x
    
    def spatial_translation(self, x: torch.Tensor, shift_range: float = 0.1) -> torch.Tensor:
        """
        Пространственное смещение
        Основано на победном решении
        """
        if random.random() > self.spatial_prob:
            return x
            
        shift_x = random.uniform(-shift_range, shift_range)
        shift_y = random.uniform(-shift_range, shift_range)
        
        # Применяем смещение к x, y координатам
        for i in range(0, x.shape[-1], 3):  # Каждая тройка (x, y, z)
            if i + 1 < x.shape[-1]:
                x[:, :, i:i+1] += shift_x
                x[:, :, i+1:i+2] += shift_y
        
        return x
    
    def feature_masking(self, x: torch.Tensor, max_mask_ratio: float = 0.1) -> torch.Tensor:
        """
        Маскирование признаков (feature masking)
        Основано на победном решении
        """
        if random.random() > self.mask_prob:
            return x
            
        feature_dim = x.shape[-1]
        mask_size = int(feature_dim * random.uniform(0, max_mask_ratio))
        
        if mask_size > 0:
            start_idx = random.randint(0, feature_dim - mask_size)
            x[:, :, start_idx:start_idx + mask_size] = 0
        
        return x
    
    def landmark_dropout(self, x: torch.Tensor, dropout_prob: float = 0.1) -> torch.Tensor:
        """
        Dropout для отдельных landmark точек
        Основано на победном решении
        """
        if random.random() > self.mask_prob:
            return x
            
        # Создаем маску для dropout
        mask = torch.bernoulli(torch.full_like(x, 1 - dropout_prob))
        return x * mask
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяем все аугментации в правильном порядке
        Основано на победном решении
        """
        # 1. Временные аугментации
        x = self.temporal_resample(x)
        x = self.temporal_masking(x)
        
        # 2. Пространственные аугментации
        x = self.spatial_rotation(x)
        x = self.spatial_scale(x)
        x = self.spatial_translation(x)
        x = self.spatial_flip(x)
        
        # 3. Шум и маскирование
        x = self.spatial_noise(x)
        x = self.feature_masking(x)
        x = self.landmark_dropout(x)
        
        return x

def test_augmentations():
    """Тест аугментаций"""
    print("🧪 Тестируем аугментации...")
    
    # Тестовые данные: (batch_size, frames, features)
    # features = landmarks * 3 (x, y, z) + motion features
    # После препроцессинга: 468 landmarks * 3 coords * 3 (pos + vel + acc) = 4212 features
    batch_size, frames, features = 2, 50, 468 * 3 * 3  # 468 landmarks * 3 coords * 3 (pos + vel + acc)
    x = torch.randn(batch_size, frames, features)
    
    augmenter = ASLAugmentations()
    
    original_shape = x.shape
    x_aug = augmenter(x.clone())
    
    print(f"   Original shape: {original_shape}")
    print(f"   Augmented shape: {x_aug.shape}")
    print(f"   Values changed: {not torch.equal(x, x_aug)}")
    
    # Проверяем изменение величины только если размеры совпадают
    if x_aug.shape == x.shape:
        print(f"   Change magnitude: {torch.mean(torch.abs(x_aug - x)):.6f}")
    else:
        print(f"   Shape changed: {x.shape} -> {x_aug.shape}")
        # Сравниваем только первые кадры, если размеры разные
        min_frames = min(x.shape[1], x_aug.shape[1])
        change_mag = torch.mean(torch.abs(x_aug[:, :min_frames] - x[:, :min_frames]))
        print(f"   Change magnitude (first {min_frames} frames): {change_mag:.6f}")
    
    # Тестируем отдельные аугментации
    print("\n   Тестируем отдельные аугментации:")
    
    # Temporal resample
    x_temp = augmenter.temporal_resample(x.clone())
    print(f"   Temporal resample: {x_temp.shape[1]} frames (было {x.shape[1]}) - размер сохранен: {x_temp.shape[1] == x.shape[1]}")
    
    # Spatial flip
    x_flip = augmenter.spatial_flip(x.clone())
    print(f"   Spatial flip: x coords changed = {torch.any(x_flip[:, :, ::3] != x[:, :, ::3])}")
    
    # Spatial rotation
    x_rot = augmenter.spatial_rotation(x.clone())
    print(f"   Spatial rotation: changed = {torch.any(x_rot != x)}")
    
    return True

if __name__ == "__main__":
    test_augmentations()
    print("✅ Аугментации готовы!")