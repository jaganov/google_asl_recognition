import pandas as pd
import json
import os
import shutil
from collections import Counter
from tqdm import tqdm
import numpy as np
import random

# Параметры
INPUT_CSV = 'manual/dataset25/train.csv'
INPUT_MAP = 'manual/dataset25/sign_to_prediction_index_map.json'
INPUT_LANDMARKS = 'manual/dataset25/train_landmark_files'

# Выходные директории
OUTPUT_BASE = 'manual/dataset25_split'
TRAIN_DIR = os.path.join(OUTPUT_BASE, 'train')
TEST_DIR = os.path.join(OUTPUT_BASE, 'test')

# Параметры разделения
TEST_SIZE = 0.2  # 20% для теста
RANDOM_STATE = 42  # Для воспроизводимости

def analyze_dataset(df):
    """Анализирует структуру датасета для правильного разделения"""
    print("=== АНАЛИЗ ДАТАСЕТА ===")
    print(f"Всего записей: {len(df)}")
    print(f"Уникальных участников: {df['participant_id'].nunique()}")
    print(f"Уникальных слов: {df['sign'].nunique()}")
    
    # Статистика по участникам
    participant_stats = df['participant_id'].value_counts()
    print(f"\nСтатистика по участникам:")
    print(f"Максимум записей от одного участника: {participant_stats.max()}")
    print(f"Минимум записей от одного участника: {participant_stats.min()}")
    print(f"Среднее записей на участника: {participant_stats.mean():.1f}")
    
    # Статистика по словам
    sign_stats = df['sign'].value_counts()
    print(f"\nСтатистика по словам:")
    print(f"Максимум записей для одного слова: {sign_stats.max()}")
    print(f"Минимум записей для одного слова: {sign_stats.min()}")
    print(f"Среднее записей на слово: {sign_stats.mean():.1f}")
    
    # Проверяем, есть ли участники с записями только одного слова
    participant_word_counts = df.groupby('participant_id')['sign'].nunique()
    single_word_participants = participant_word_counts[participant_word_counts == 1]
    print(f"\nУчастников с одним словом: {len(single_word_participants)}")
    
    return participant_stats, sign_stats, participant_word_counts

def create_stratified_split(df, test_size=0.2, random_state=42):
    """
    Создает стратифицированное разделение с учетом особенностей жестов:
    1. Разделяем по участникам, чтобы избежать утечки данных
    2. Обеспечиваем баланс по словам в train и test
    3. Учитываем, что это анимации жестов от одних и тех же людей
    """
    print("\n=== СОЗДАНИЕ РАЗДЕЛЕНИЯ ===")
    
    # Получаем уникальных участников
    unique_participants = df['participant_id'].unique()
    print(f"Уникальных участников для разделения: {len(unique_participants)}")
    
    # Создаем стратификацию по количеству слов у участника
    participant_word_counts = df.groupby('participant_id')['sign'].nunique()
    
    # Группируем участников по количеству слов (для стратификации)
    word_count_bins = pd.cut(participant_word_counts, bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # Создаем DataFrame для разделения
    split_df = pd.DataFrame({
        'participant_id': unique_participants,
        'word_count': participant_word_counts.values,
        'word_count_bin': word_count_bins.values
    })
    
    # Устанавливаем seed для воспроизводимости
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Простое разделение участников на train и test
    participants_list = unique_participants.tolist()
    random.shuffle(participants_list)
    
    # Разделяем по пропорции
    split_idx = int(len(participants_list) * (1 - test_size))
    train_participants = participants_list[:split_idx]
    test_participants = participants_list[split_idx:]
    
    print(f"Участников в train: {len(train_participants)}")
    print(f"Участников в test: {len(test_participants)}")
    
    # Разделяем данные
    train_df = df[df['participant_id'].isin(train_participants)].copy()
    test_df = df[df['participant_id'].isin(test_participants)].copy()
    
    print(f"Записей в train: {len(train_df)}")
    print(f"Записей в test: {len(test_df)}")
    
    # Проверяем баланс по словам
    print("\n=== ПРОВЕРКА БАЛАНСА ===")
    train_signs = train_df['sign'].value_counts()
    test_signs = test_df['sign'].value_counts()
    
    print("Слова в train:")
    for sign, count in train_signs.items():
        print(f"  {sign}: {count}")
    
    print("\nСлова в test:")
    for sign, count in test_signs.items():
        print(f"  {sign}: {count}")
    
    # Проверяем, что все слова присутствуют в обоих наборах
    train_sign_set = set(train_signs.index)
    test_sign_set = set(test_signs.index)
    all_signs = set(df['sign'].unique())
    
    missing_in_train = all_signs - train_sign_set
    missing_in_test = all_signs - test_sign_set
    
    if missing_in_train:
        print(f"\n⚠️  Слова отсутствуют в train: {missing_in_train}")
    if missing_in_test:
        print(f"⚠️  Слова отсутствуют в test: {missing_in_test}")
    
    if not missing_in_train and not missing_in_test:
        print("\n✅ Все слова присутствуют в train и test")
    
    return train_df, test_df

def copy_files_for_split(df, output_dir, split_name):
    """Копирует файлы для указанного набора данных"""
    print(f"\n=== КОПИРОВАНИЕ ФАЙЛОВ ДЛЯ {split_name.upper()} ===")
    
    # Создаем директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем уникальные пути
    unique_paths = df['path'].unique()
    copied_count = 0
    skipped_count = 0
    
    with tqdm(total=len(unique_paths), desc=f"Копирование {split_name}") as pbar:
        for rel_path in unique_paths:
            src = os.path.join('manual/dataset25', rel_path)
            dst = os.path.join(output_dir, rel_path)
            dst_dir = os.path.dirname(dst)
            
            try:
                os.makedirs(dst_dir, exist_ok=True)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f'Ошибка при копировании {rel_path}: {e}')
                skipped_count += 1
            pbar.update(1)
    
    print(f'Скопировано файлов для {split_name}: {copied_count}')
    print(f'Пропущено (уже существуют): {skipped_count}')
    return copied_count, skipped_count

def main():
    """Основная функция для разделения датасета"""
    print("🚀 НАЧАЛО РАЗДЕЛЕНИЯ ДАТАСЕТА НА TRAIN И TEST")
    print("=" * 60)
    
    # 1. Загружаем данные
    print('Загружаем данные...')
    df = pd.read_csv(INPUT_CSV)
    
    # 2. Анализируем структуру
    participant_stats, sign_stats, participant_word_counts = analyze_dataset(df)
    
    # 3. Создаем разделение
    train_df, test_df = create_stratified_split(df, TEST_SIZE, RANDOM_STATE)
    
    # 4. Создаем выходные директории
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # 5. Сохраняем CSV файлы
    train_csv = os.path.join(TRAIN_DIR, 'train.csv')
    test_csv = os.path.join(TEST_DIR, 'train.csv')  # Используем то же имя для совместимости
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f'\nСохранено train.csv: {train_csv}')
    print(f'Сохранено test.csv: {test_csv}')
    
    # 6. Копируем sign_to_prediction_index_map.json в обе директории
    for output_dir in [TRAIN_DIR, TEST_DIR]:
        shutil.copy2(INPUT_MAP, os.path.join(output_dir, 'sign_to_prediction_index_map.json'))
    
    # 7. Копируем файлы данных
    copy_files_for_split(train_df, TRAIN_DIR, 'train')
    copy_files_for_split(test_df, TEST_DIR, 'test')
    
    # 8. Создаем итоговую статистику
    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    
    print(f"Исходный датасет: {len(df)} записей")
    print(f"Train набор: {len(train_df)} записей ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test набор: {len(test_df)} записей ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nУчастников в train: {train_df['participant_id'].nunique()}")
    print(f"Участников в test: {test_df['participant_id'].nunique()}")
    
    print(f"\nСлов в train: {train_df['sign'].nunique()}")
    print(f"Слов в test: {test_df['sign'].nunique()}")
    
    # Проверяем минимальное количество записей на слово
    train_min = train_df['sign'].value_counts().min()
    test_min = test_df['sign'].value_counts().min()
    
    print(f"\nМинимум записей на слово в train: {train_min}")
    print(f"Минимум записей на слово в test: {test_min}")
    
    if train_min < 10 or test_min < 5:
        print("⚠️  ВНИМАНИЕ: Некоторые слова имеют мало записей!")
        print("   Рекомендуется проверить качество разделения.")
    
    print("\n✅ РАЗДЕЛЕНИЕ ЗАВЕРШЕНО!")
    print(f"📁 Train данные: {TRAIN_DIR}")
    print(f"📁 Test данные: {TEST_DIR}")

if __name__ == "__main__":
    main() 