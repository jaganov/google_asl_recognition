import torch
from torch import nn
import numpy as np
import math
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)
torch.manual_seed(2)  # we set up a seed so that your output matches ours although the initialization is random.
dtype = torch.float
dtype_long = torch.long
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def load_dataset(data_dir: str = "dataset25_split", max_len: int = 384, max_samples: Optional[int] = None):
    """
    Загружает Google ASL Signs dataset для тренировки (оптимизированная версия)
    
    Args:
        data_dir: Путь к папке с данными (dataset25_split)
        max_len: Максимальная длина последовательности кадров
        max_samples: Максимальное количество образцов для загрузки (None = все)
    
    Returns:
        train_data: Список тензоров landmarks для тренировки
        train_labels: Список меток для тренировки  
        test_data: Список тензоров landmarks для тестирования
        test_labels: Список меток для тестирования
        sign_mapping: Словарь маппинга знаков к индексам
        classes: Список классов (знаков)
    """
    print("🚀 Загрузка Google ASL Signs dataset (оптимизированная версия)...")
    
    data_path = Path(data_dir)
    
    # Проверяем существование директорий
    if not (data_path / "train").exists():
        raise FileNotFoundError(f"Директория {data_path / 'train'} не найдена")
    if not (data_path / "test").exists():
        raise FileNotFoundError(f"Директория {data_path / 'test'} не найдена")
    
    # Загружаем маппинг знаков
    sign_mapping_path = data_path / "train" / "sign_to_prediction_index_map.json"
    if not sign_mapping_path.exists():
        raise FileNotFoundError(f"Файл {sign_mapping_path} не найден")
    
    with open(sign_mapping_path, 'r') as f:
        sign_mapping = json.load(f)
    
    print(f"📊 Загружено {len(sign_mapping)} знаков")
    
    # Загружаем train данные
    print("\n📁 Загрузка тренировочных данных...")
    train_csv_path = data_path / "train" / "train.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Файл {train_csv_path} не найден")
    
    train_df = pd.read_csv(train_csv_path)
    print(f"   Найдено {len(train_df)} записей в train.csv")
    
    # Ограничиваем количество образцов если указано
    if max_samples and len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"   Ограничено до {max_samples} образцов для тестирования")
    
    train_data, train_labels = _load_sequences_optimized(train_df, data_path / "train", sign_mapping, max_len)
    
    # Загружаем test данные
    print("\n📁 Загрузка тестовых данных...")
    test_csv_path = data_path / "test" / "train.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Файл {test_csv_path} не найден")
    
    test_df = pd.read_csv(test_csv_path)
    print(f"   Найдено {len(test_df)} записей в test/train.csv")
    
    # Ограничиваем количество образцов если указано
    if max_samples and len(test_df) > max_samples:
        test_df = test_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"   Ограничено до {max_samples} образцов для тестирования")
    
    test_data, test_labels = _load_sequences_optimized(test_df, data_path / "test", sign_mapping, max_len)
    
    # Создаем список классов
    classes = list(sign_mapping.keys())
    
    print(f"\n✅ Загрузка завершена!")
    print(f"   Тренировочных последовательностей: {len(train_data)}")
    print(f"   Тестовых последовательностей: {len(test_data)}")
    print(f"   Количество классов: {len(classes)}")
    print(f"   Максимальная длина последовательности: {max_len}")
    
    return train_data, train_labels, test_data, test_labels, sign_mapping, classes

def _load_sequences_optimized(df: pd.DataFrame, data_dir: Path, sign_mapping: Dict[str, int], max_len: int) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Оптимизированная загрузка последовательностей landmarks
    """
    sequences = []
    labels = []
    
    total_files = len(df)
    successful_loads = 0
    failed_loads = 0
    
    print(f"   Начинаем загрузку {total_files} файлов...")
    
    for idx, row in df.iterrows():
        # Прогресс каждые 50 файлов
        if idx % 50 == 0:
            print(f"   Прогресс: {idx}/{total_files} (успешно: {successful_loads}, ошибок: {failed_loads})")
        
        file_path = data_dir / row['path']
        sign = row['sign']
        
        # Проверяем существование файла
        if not file_path.exists():
            print(f"   ⚠️ Файл не найден: {file_path}")
            failed_loads += 1
            continue
        
        try:
            # Загружаем parquet файл с ограничением памяти
            landmarks_df = pd.read_parquet(file_path, engine='pyarrow')
            
            # Проверяем, что файл не пустой
            if len(landmarks_df) == 0:
                print(f"   ⚠️ Пустой файл: {file_path}")
                failed_loads += 1
                continue
            
            # Проверяем наличие необходимых колонок
            required_columns = ['frame', 'landmark_index', 'x', 'y', 'z']
            if not all(col in landmarks_df.columns for col in required_columns):
                print(f"   ⚠️ Неправильная структура файла: {file_path}")
                failed_loads += 1
                continue
            
            # Преобразуем в тензор landmarks
            sequence_tensor = _parquet_to_tensor_optimized(landmarks_df, max_len)
            
            # Проверяем, что тензор не пустой
            if sequence_tensor.shape[0] == 0:
                print(f"   ⚠️ Пустая последовательность: {file_path}")
                failed_loads += 1
                continue
            
            # Получаем метку
            label = sign_mapping[sign]
            
            sequences.append(sequence_tensor)
            labels.append(label)
            successful_loads += 1
            
        except Exception as e:
            print(f"   ❌ Ошибка при загрузке {file_path}: {str(e)[:100]}...")
            failed_loads += 1
            continue
    
    print(f"   Загрузка завершена: успешно {successful_loads}, ошибок {failed_loads}")
    
    return sequences, labels

def _parquet_to_tensor_optimized(landmarks_df: pd.DataFrame, max_len: int) -> torch.Tensor:
    """
    Оптимизированное преобразование parquet DataFrame в тензор landmarks
    """
    try:
        # Получаем уникальные кадры и сортируем их
        frames = sorted(landmarks_df['frame'].unique())
        
        # Ограничиваем количество кадров
        frames = frames[:max_len]
        
        if len(frames) == 0:
            return torch.zeros((0, 0, 3), dtype=torch.float32)
        
        # Создаем правильную структуру landmarks с учетом типов
        # Face: 468 landmarks (0-467)
        # Pose: 33 landmarks (468-500)
        # Left Hand: 21 landmarks (501-521)
        # Right Hand: 21 landmarks (522-542)
        # Всего: 543 landmarks
        
        total_landmarks = 543
        landmarks_tensor = torch.zeros((len(frames), total_landmarks, 3), dtype=torch.float32)
        
        # Оптимизированное заполнение тензора
        for frame_idx, frame in enumerate(frames):
            # Данные для текущего кадра
            frame_data = landmarks_df[landmarks_df['frame'] == frame]
            
            # Создаем словарь для быстрого поиска с учетом типов
            frame_dict = {}
            for _, row in frame_data.iterrows():
                landmark_idx = row['landmark_index']
                landmark_type = row['type']
                
                # Определяем правильный индекс в тензоре
                if landmark_type == 'face':
                    tensor_idx = landmark_idx  # 0-467
                elif landmark_type == 'pose':
                    tensor_idx = 468 + landmark_idx  # 468-500
                elif landmark_type == 'left_hand':
                    tensor_idx = 501 + landmark_idx  # 501-521
                elif landmark_type == 'right_hand':
                    tensor_idx = 522 + landmark_idx  # 522-542
                else:
                    continue  # Пропускаем неизвестные типы
                
                frame_dict[tensor_idx] = [row['x'], row['y'], row['z']]
            
            # Заполняем тензор
            for tensor_idx in range(total_landmarks):
                if tensor_idx in frame_dict:
                    coords = frame_dict[tensor_idx]
                    landmarks_tensor[frame_idx, tensor_idx] = torch.tensor(coords, dtype=torch.float32)
        
        return landmarks_tensor
        
    except Exception as e:
        print(f"   Ошибка при преобразовании в тензор: {str(e)[:100]}...")
        return torch.zeros((0, 0, 3), dtype=torch.float32)

def get_dataset_statistics(train_data: List[torch.Tensor], test_data: List[torch.Tensor], sign_mapping: Dict[str, int]):
    """
    Выводит статистику загруженного датасета
    """
    print("\n📊 Статистика датасета:")
    
    # Статистика по длине последовательностей
    train_lengths = [seq.shape[0] for seq in train_data]
    test_lengths = [seq.shape[0] for seq in test_data]
    
    print(f"   Тренировочные последовательности:")
    print(f"     Минимум кадров: {min(train_lengths)}")
    print(f"     Максимум кадров: {max(train_lengths)}")
    print(f"     Среднее кадров: {np.mean(train_lengths):.1f}")
    print(f"     Медиана кадров: {np.median(train_lengths):.1f}")
    
    print(f"   Тестовые последовательности:")
    print(f"     Минимум кадров: {min(test_lengths)}")
    print(f"     Максимум кадров: {max(test_lengths)}")
    print(f"     Среднее кадров: {np.mean(test_lengths):.1f}")
    print(f"     Медиана кадров: {np.median(test_lengths):.1f}")
    
    # Статистика по landmarks
    if train_data:
        sample_seq = train_data[0]
        print(f"   Размерность landmarks: {sample_seq.shape[1]} точек")
        print(f"   Координаты: x, y, z (3 измерения)")
        print(f"   Структура landmarks:")
        print(f"     - Face: 468 точек (0-467)")
        print(f"     - Pose: 33 точки (468-500)")
        print(f"     - Left Hand: 21 точка (501-521)")
        print(f"     - Right Hand: 21 точка (522-542)")
        print(f"     - Всего: 543 точки")
    
    # Статистика по классам
    print(f"   Количество классов: {len(sign_mapping)}")
    print(f"   Классы: {list(sign_mapping.keys())}")

# Пример использования
if __name__ == "__main__":
    # Загружаем датасет с ограничением для тестирования
    print("🧪 Тестируем загрузку с ограниченным количеством образцов...")
    train_data, train_labels, test_data, test_labels, sign_mapping, classes = load_dataset(max_samples=100)
    
    # Выводим статистику
    get_dataset_statistics(train_data, test_data, sign_mapping)
    
    print(f"\n🎯 Готово к тренировке!")
    print(f"   Используйте train_data и train_labels для тренировки")
    print(f"   Используйте test_data и test_labels для валидации")
    print(f"   Каждая последовательность - это анимация жеста MediaPipe")
    print(f"\n💡 Для загрузки всего датасета используйте: load_dataset(max_samples=None)")





