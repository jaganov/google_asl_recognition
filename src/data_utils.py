# data_utils.py
import pandas as pd
import numpy as np
import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class ASLDataAnalyzer:
    """Анализатор данных Google ASL Signs"""
    
    def __init__(self, data_dir: str = "../data/google_asl_signs"):
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.sign_mapping = None
        self.load_data()
    
    def load_data(self):
        """Загружает основные данные"""
        print("📊 Загрузка данных...")
        
        # Загружаем train.csv
        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        print(f"   Загружено {len(self.train_df)} записей")
        
        # Загружаем маппинг знаков
        with open(self.data_dir / "sign_to_prediction_index_map.json", 'r') as f:
            self.sign_mapping = json.load(f)
        print(f"   Загружено {len(self.sign_mapping)} знаков")
    
    def analyze_dataset_statistics(self):
        """Анализ общей статистики датасета"""
        print("\n📈 ОБЩАЯ СТАТИСТИКА ДАТАСЕТА")
        print("=" * 50)
        
        # Основная статистика
        print(f"📊 Общая статистика:")
        print(f"   Всего записей: {len(self.train_df):,}")
        print(f"   Уникальных знаков: {self.train_df['sign'].nunique()}")
        print(f"   Уникальных участников: {self.train_df['participant_id'].nunique()}")
        print(f"   Уникальных последовательностей: {self.train_df['sequence_id'].nunique()}")
        
        # Статистика знаков
        sign_counts = self.train_df['sign'].value_counts()
        print(f"\n🎯 Статистика знаков:")
        print(f"   Самый частый знак: {sign_counts.index[0]} ({sign_counts.iloc[0]} записей)")
        print(f"   Самый редкий знак: {sign_counts.index[-1]} ({sign_counts.iloc[-1]} записей)")
        print(f"   Среднее количество записей на знак: {sign_counts.mean():.1f}")
        print(f"   Медиана записей на знак: {sign_counts.median():.1f}")
        
        # Статистика участников
        participant_counts = self.train_df['participant_id'].value_counts()
        print(f"\n👥 Статистика участников:")
        print(f"   Самый активный участник: {participant_counts.index[0]} ({participant_counts.iloc[0]} записей)")
        print(f"   Среднее количество записей на участника: {participant_counts.mean():.1f}")
        print(f"   Медиана записей на участника: {participant_counts.median():.1f}")
        
        return {
            'sign_counts': sign_counts,
            'participant_counts': participant_counts
        }
    
    def analyze_landmark_files(self, sample_size: int = 100):
        """Анализ файлов с landmarks"""
        print(f"\n🎯 Анализ landmark файлов (выборка из {sample_size} файлов)...")
        
        # Выбираем случайную выборку файлов для анализа
        sample_df = self.train_df.sample(n=min(sample_size, len(self.train_df)), random_state=42)
        
        frame_counts = []
        landmark_counts = []
        file_sizes = []
        missing_data_ratios = []
        landmark_types = []
        
        for idx, row in sample_df.iterrows():
            file_path = self.data_dir / row['path']
            
            if file_path.exists():
                # Размер файла
                file_size = file_path.stat().st_size
                file_sizes.append(file_size)
                
                try:
                    # Загружаем parquet файл
                    landmarks_df = pd.read_parquet(file_path)
                    
                    # Анализируем структуру
                    frame_counts.append(len(landmarks_df['frame'].unique()))
                    
                    # Типы landmarks
                    types = landmarks_df['type'].unique()
                    landmark_types.extend(types)
                    
                    # Количество уникальных landmarks
                    unique_landmarks = landmarks_df['landmark_index'].nunique()
                    landmark_counts.append(unique_landmarks)
                    
                    # Проверка на пропущенные данные
                    missing_ratio = landmarks_df.isnull().sum().sum() / landmarks_df.size
                    missing_data_ratios.append(missing_ratio)
                    
                except Exception as e:
                    print(f"   Ошибка при чтении {file_path}: {e}")
                    continue
        
        if frame_counts:
            print(f"   Статистика кадров:")
            print(f"     Минимум: {min(frame_counts)} кадров")
            print(f"     Максимум: {max(frame_counts)} кадров")
            print(f"     Среднее: {np.mean(frame_counts):.1f} кадров")
            print(f"     Медиана: {np.median(frame_counts):.1f} кадров")
        
        if landmark_counts:
            print(f"   Статистика landmarks:")
            print(f"     Среднее количество точек: {np.mean(landmark_counts):.1f}")
            print(f"     Минимум точек: {min(landmark_counts)}")
            print(f"     Максимум точек: {max(landmark_counts)}")
        
        if file_sizes:
            print(f"   Статистика размеров файлов:")
            print(f"     Минимум: {min(file_sizes)} байт")
            print(f"     Максимум: {max(file_sizes)} байт")
            print(f"     Среднее: {np.mean(file_sizes):.1f} байт")
            print(f"     Медиана: {np.median(file_sizes):.1f} байт")
        
        if missing_data_ratios:
            print(f"   Качество данных:")
            print(f"     Среднее количество пропущенных данных: {np.mean(missing_data_ratios):.3%}")
            print(f"     Максимальное количество пропущенных данных: {max(missing_data_ratios):.3%}")
        
        # Анализ типов landmarks
        type_counter = Counter(landmark_types)
        print(f"   Типы landmarks:")
        for landmark_type, count in type_counter.most_common():
            print(f"     {landmark_type}: {count} файлов")
        
        return {
            'frame_counts': frame_counts,
            'landmark_counts': landmark_counts,
            'file_sizes': file_sizes,
            'missing_data_ratios': missing_data_ratios,
            'landmark_types': type_counter
        }
    
    def create_balanced_splits(self, 
                              train_ratio: float = 0.8,
                              val_ratio: float = 0.1,
                              test_ratio: float = 0.1,
                              min_samples_per_class: int = 5):
        """Создает сбалансированные сплиты train/val/test"""
        print(f"\n✂️ Создание сбалансированных сплитов...")
        
        # Проверяем соотношение
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Сумма долей должна быть 1.0"
        
        # Фильтруем классы с достаточным количеством образцов
        sign_counts = self.train_df['sign'].value_counts()
        valid_signs = sign_counts[sign_counts >= min_samples_per_class].index
        
        print(f"   Знаков с >= {min_samples_per_class} образцами: {len(valid_signs)}")
        
        # Фильтруем данные
        filtered_df = self.train_df[self.train_df['sign'].isin(valid_signs)].copy()
        print(f"   Записей после фильтрации: {len(filtered_df)}")
        
        # Создаем сплиты
        splits = {}
        
        for sign in valid_signs:
            sign_data = filtered_df[filtered_df['sign'] == sign]
            n_samples = len(sign_data)
            
            # Вычисляем размеры сплитов
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            n_test = n_samples - n_train - n_val
            
            # Перемешиваем индексы
            indices = sign_data.index.tolist()
            np.random.shuffle(indices)
            
            # Разделяем на сплиты
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
            
            # Сохраняем индексы
            if 'train' not in splits:
                splits['train'] = []
                splits['val'] = []
                splits['test'] = []
            
            splits['train'].extend(train_indices)
            splits['val'].extend(val_indices)
            splits['test'].extend(test_indices)
        
        # Создаем DataFrame для каждого сплита
        train_df = filtered_df.loc[splits['train']].reset_index(drop=True)
        val_df = filtered_df.loc[splits['val']].reset_index(drop=True)
        test_df = filtered_df.loc[splits['test']].reset_index(drop=True)
        
        print(f"   Размеры сплитов:")
        print(f"     Train: {len(train_df)} записей")
        print(f"     Val: {len(val_df)} записей")
        print(f"     Test: {len(test_df)} записей")
        
        # Сохраняем сплиты
        splits_dir = self.data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        train_df.to_csv(splits_dir / "train.csv", index=False)
        val_df.to_csv(splits_dir / "val.csv", index=False)
        test_df.to_csv(splits_dir / "test.csv", index=False)
        
        print(f"   Сплиты сохранены в {splits_dir}")
        
        return train_df, val_df, test_df
    
    def get_class_weights(self, split_df: pd.DataFrame) -> torch.Tensor:
        """Вычисляет веса классов для сбалансированного обучения"""
        sign_counts = split_df['sign'].value_counts()
        
        # Вычисляем веса (обратно пропорционально частоте)
        total_samples = len(split_df)
        class_weights = total_samples / (len(sign_counts) * sign_counts)
        
        # Сортируем по индексам классов
        class_weights = class_weights.sort_index()
        
        return torch.tensor(class_weights.values, dtype=torch.float32)
    
    def visualize_data_distribution(self, save_dir: str = "exploration_output"):
        """Визуализирует распределение данных"""
        print(f"\n📊 Создание визуализаций...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 1. Распределение знаков
        sign_counts = self.train_df['sign'].value_counts()
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sign_counts.head(20).plot(kind='bar')
        plt.title('Топ-20 самых частых знаков')
        plt.xlabel('Знак')
        plt.ylabel('Количество записей')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.hist(sign_counts.values, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Распределение частоты знаков')
        plt.xlabel('Количество записей на знак')
        plt.ylabel('Количество знаков')
        
        plt.subplot(2, 2, 3)
        participant_counts = self.train_df['participant_id'].value_counts()
        plt.hist(participant_counts.values, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Распределение записей на участника')
        plt.xlabel('Количество записей')
        plt.ylabel('Количество участников')
        
        plt.subplot(2, 2, 4)
        plt.boxplot([sign_counts.values, participant_counts.values], 
                   labels=['Знаки', 'Участники'])
        plt.title('Box plot распределений')
        plt.ylabel('Количество записей')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'data_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Сохранена визуализация: data_distribution.png")

def main():
    """Основная функция анализа"""
    print("🔍 АНАЛИЗ GOOGLE ASL SIGNS ДАТАСЕТА")
    print("=" * 50)
    
    # Создаем анализатор
    analyzer = ASLDataAnalyzer()
    
    # Анализируем данные
    stats = analyzer.analyze_dataset_statistics()
    landmark_stats = analyzer.analyze_landmark_files(sample_size=200)
    
    # Создаем визуализации
    analyzer.visualize_data_distribution()
    
    # Создаем сбалансированные сплиты
    train_df, val_df, test_df = analyzer.create_balanced_splits()
    
    print("\n✅ Анализ данных завершен!")

if __name__ == "__main__":
    main() 