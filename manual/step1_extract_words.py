import pandas as pd
import json
import os
import shutil
from collections import Counter
from tqdm import tqdm

# Параметры
INPUT_CSV = 'data/google_asl_signs/train.csv'
INPUT_MAP = 'data/google_asl_signs/sign_to_prediction_index_map.json'
INPUT_LANDMARKS = 'data/google_asl_signs/train_landmark_files'
OUTPUT_DIR = 'manual/dataset25'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'train.csv')
OUTPUT_MAP = os.path.join(OUTPUT_DIR, 'sign_to_prediction_index_map.json')
OUTPUT_LANDMARKS = os.path.join(OUTPUT_DIR, 'train_landmark_files')

# Список базовых слов, которые должен знать каждый (в порядке важности)
BASIC_WORDS = [
    # Приветствия и основные слова
    'hello', 'yes,no', 'please', 'thankyou', 'sorry', 'goodbye', 'bye',
    
    # Личные местоимения
    'i', 'you', 'he', 'she', 'we', 'they', 'me', 'my', 'your', 'his,her',
    
    # Семья
    'mom', 'dad', 'baby', 'boy', 'girl', 'man', 'woman', 'child', 'family',
    
    # Основные действия
    'eat', 'drink', 'sleep', 'walk', 'un', 'sit', 'stand', 'come', 'go', 'stop',
    
    # Эмоции и состояния
    'happy', 'sad', 'angry', 'tired', 'hungry', 'thirsty', 'sick', 'ood', 'bad',
    
    # Цвета
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink',
    
    # Числа
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    
    # Время
    'today', 'tomorrow', 'yesterday', 'morning', 'afternoon', 'night', 'time',
    
    # Основные предметы
    'water', 'food', 'house', 'car', 'book', 'phone', 'money', 'help'
]

def get_available_basic_words(df, basic_words_list):
    """Получает список базовых слов, которые есть в датасете"""
    available_words = []
    missing_words = []
    
    for word in basic_words_list:
        if word in df['sign'].values:
            available_words.append(word)
        else:
            missing_words.append(word)
    
    print(f"Найдено базовых слов в датасете: {len(available_words)}")
    if missing_words:
        print(f"Отсутствуют в датасете: {missing_words}")
    return available_words

def get_top_words_by_importance(df, basic_words, target_count=50):
    """Выбирает слова по важности: сначала базовые, потом по частоте"""
    available_basic = get_available_basic_words(df, basic_words)
    
    # Берем все доступные базовые слова
    selected_words = available_basic[:target_count]
    
    # Если базовых слов меньше 50, добавляем самые частые из оставшихся
    if len(selected_words) < target_count:
        remaining_words = df[~df['sign'].isin(selected_words)]['sign'].value_counts()
        additional_words = remaining_words.head(target_count - len(selected_words)).index.tolist()
        selected_words.extend(additional_words)
    
    return selected_words[:target_count]

# 1. Загрузка данных
print('Загружаем train.csv...')
df = pd.read_csv(INPUT_CSV)
print(f'Всего записей: {len(df)}')
print(f'Уникальных слов: {df["sign"].nunique()}')

# 2. Выбираем 25 важных слов
print('Выбираем 25 важных слов...')
top_25 = get_top_words_by_importance(df, BASIC_WORDS, 25)
print(f'Выбранные слова ({len(top_25)}):')
for i, word in enumerate(top_25):
    count = len(df[df['sign'] == word])
    print(f'{i:2d} {word:15s} ({count:5d} записей)')

# 3. Фильтруем строки только с этими словами
filtered_df = df[df['sign'].isin(top_25)].copy()
print(f'\nСтрок после фильтрации: {len(filtered_df)}')

# 4. Сохраняем новый train.csv
os.makedirs(OUTPUT_DIR, exist_ok=True)
filtered_df.to_csv(OUTPUT_CSV, index=False)
print(f'Сохранено: {OUTPUT_CSV}')

# 5. Создаем новый sign_to_prediction_index_map.json
with open(INPUT_MAP, 'r') as f:
    full_map = json.load(f)

new_map = {sign: i for i, sign in enumerate(top_25)}
with open(OUTPUT_MAP, 'w') as f:
    json.dump(new_map, f, indent=2)
print(f'Сохранено: {OUTPUT_MAP}')

# 6. Копируем parquet-файлы с прогресс-баром
os.makedirs(OUTPUT_LANDMARKS, exist_ok=True)
print('\nКопируем parquet-файлы...')

# Получаем уникальные пути для копирования
unique_paths = filtered_df['path'].unique()
copied_count = 0
skipped_count = 0

with tqdm(total=len(unique_paths), desc="Копирование файлов") as pbar:
    for rel_path in unique_paths:
        src = os.path.join('data/google_asl_signs', rel_path)
        dst = os.path.join(OUTPUT_DIR, rel_path)
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

print('\nГотово!')
print(f'Скопировано файлов: {copied_count}')
print(f'Пропущено (уже существуют): {skipped_count}')
print(f'Всего обработано: {len(unique_paths)}')

def create_train_test_split():
    """Создает разделение на train и test датасеты"""
    print('\n' + '='*60)
    print('🚀 СОЗДАНИЕ РАЗДЕЛЕНИЯ НА TRAIN И TEST')
    print('='*60)
    
    # Импортируем необходимые модули
    import numpy as np
    import random
    
    # Параметры
    SPLIT_OUTPUT_BASE = 'manual/dataset25_split'
    TRAIN_DIR = os.path.join(SPLIT_OUTPUT_BASE, 'train')
    TEST_DIR = os.path.join(SPLIT_OUTPUT_BASE, 'test')
    TEST_SIZE = 0.2  # 20% для теста
    RANDOM_STATE = 42
    
    # Загружаем данные
    print('Загружаем данные для разделения...')
    df = pd.read_csv(OUTPUT_CSV)
    
    # Анализируем структуру
    print(f"Всего записей: {len(df)}")
    print(f"Уникальных участников: {df['participant_id'].nunique()}")
    print(f"Уникальных слов: {df['sign'].nunique()}")
    
    # Создаем стратифицированное разделение по участникам
    unique_participants = df['participant_id'].unique()
    participant_word_counts = df.groupby('participant_id')['sign'].nunique()
    
    # Группируем участников по количеству слов для стратификации
    word_count_bins = pd.cut(participant_word_counts, bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    split_df = pd.DataFrame({
        'participant_id': unique_participants,
        'word_count': participant_word_counts.values,
        'word_count_bin': word_count_bins.values
    })
    
    # Устанавливаем seed для воспроизводимости
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    # Простое разделение участников на train и test
    participants_list = unique_participants.tolist()
    random.shuffle(participants_list)
    
    # Разделяем по пропорции
    split_idx = int(len(participants_list) * (1 - TEST_SIZE))
    train_participants = participants_list[:split_idx]
    test_participants = participants_list[split_idx:]
    
    # Разделяем данные
    train_df = df[df['participant_id'].isin(train_participants)].copy()
    test_df = df[df['participant_id'].isin(test_participants)].copy()
    
    print(f"Участников в train: {len(train_participants)}")
    print(f"Участников в test: {len(test_participants)}")
    print(f"Записей в train: {len(train_df)}")
    print(f"Записей в test: {len(test_df)}")
    
    # Создаем директории
    os.makedirs(SPLIT_OUTPUT_BASE, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Сохраняем CSV файлы
    train_csv = os.path.join(TRAIN_DIR, 'train.csv')
    test_csv = os.path.join(TEST_DIR, 'train.csv')
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # Копируем sign_to_prediction_index_map.json
    for output_dir in [TRAIN_DIR, TEST_DIR]:
        shutil.copy2(OUTPUT_MAP, os.path.join(output_dir, 'sign_to_prediction_index_map.json'))
    
    # Копируем файлы данных
    def copy_files_for_split(split_df, output_dir, split_name):
        unique_paths = split_df['path'].unique()
        copied_count = 0
        
        print(f'\nКопируем файлы для {split_name}...')
        with tqdm(total=len(unique_paths), desc=f"Копирование {split_name}") as pbar:
            for rel_path in unique_paths:
                src = os.path.join(OUTPUT_DIR, rel_path)
                dst = os.path.join(output_dir, rel_path)
                dst_dir = os.path.dirname(dst)
                
                try:
                    os.makedirs(dst_dir, exist_ok=True)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        copied_count += 1
                except Exception as e:
                    print(f'Ошибка при копировании {rel_path}: {e}')
                pbar.update(1)
        
        print(f'Скопировано файлов для {split_name}: {copied_count}')
        return copied_count
    
    copy_files_for_split(train_df, TRAIN_DIR, 'train')
    copy_files_for_split(test_df, TEST_DIR, 'test')
    
    # Проверяем баланс
    print('\n=== ПРОВЕРКА БАЛАНСА ===')
    train_signs = train_df['sign'].value_counts()
    test_signs = test_df['sign'].value_counts()
    
    all_signs = set(df['sign'].unique())
    train_sign_set = set(train_signs.index)
    test_sign_set = set(test_signs.index)
    
    missing_in_train = all_signs - train_sign_set
    missing_in_test = all_signs - test_sign_set
    
    if not missing_in_train and not missing_in_test:
        print('✅ Все слова присутствуют в train и test')
    else:
        if missing_in_train:
            print(f'⚠️  Слова отсутствуют в train: {missing_in_train}')
        if missing_in_test:
            print(f'⚠️  Слова отсутствуют в test: {missing_in_test}')
    
    print(f'\n✅ РАЗДЕЛЕНИЕ ЗАВЕРШЕНО!')
    print(f'📁 Train данные: {TRAIN_DIR}')
    print(f'📁 Test данные: {TEST_DIR}')
    print(f'📊 Train: {len(train_df)} записей ({len(train_df)/len(df)*100:.1f}%)')
    print(f'📊 Test: {len(test_df)} записей ({len(test_df)/len(df)*100:.1f}%)')

# Если нужно создать разделение, раскомментируйте следующую строку:
# create_train_test_split() 