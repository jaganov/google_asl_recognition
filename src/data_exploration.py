# data_exploration.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']

DIR_DATASET = '../data/google_asl_signs/'

def load_sign_mapping():
    """Загрузка маппинга знаков к индексам"""
    with open(os.path.join(DIR_DATASET, 'sign_to_prediction_index_map.json'), 'r') as f:
        sign_mapping = json.load(f)
    return sign_mapping

def load_train_data(sample_size=None):
    """Загрузка данных из train.csv"""
    print("📊 Загрузка train.csv...")
    df = pd.read_csv(os.path.join(DIR_DATASET, 'train.csv'))
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"   Загружено {len(df)} записей")
    return df

def analyze_signs_distribution(df, sign_mapping):
    """Анализ распределения знаков"""
    print("\n📈 Анализ распределения знаков...")
    
    # Подсчет частоты каждого знака
    sign_counts = df['sign'].value_counts()
    
    print(f"   Всего уникальных знаков: {len(sign_counts)}")
    print(f"   Самые частые знаки:")
    for i, (sign, count) in enumerate(sign_counts.head(10).items()):
        print(f"     {i+1}. {sign}: {count} записей")
    
    print(f"   Самые редкие знаки:")
    for i, (sign, count) in enumerate(sign_counts.tail(10).items()):
        print(f"     {i+1}. {sign}: {count} записей")
    
    # Визуализация распределения
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 2, 1)
    sign_counts.head(20).plot(kind='bar')
    plt.title('Топ-20 самых частых знаков')
    plt.xlabel('Знак')
    plt.ylabel('Количество записей')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sign_counts.tail(20).plot(kind='bar')
    plt.title('20 самых редких знаков')
    plt.xlabel('Знак')
    plt.ylabel('Количество записей')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    plt.hist(sign_counts.values, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Распределение частоты знаков')
    plt.xlabel('Количество записей на знак')
    plt.ylabel('Количество знаков')
    
    plt.subplot(2, 2, 4)
    plt.boxplot(sign_counts.values)
    plt.title('Box plot частоты знаков')
    plt.ylabel('Количество записей на знак')
    
    plt.tight_layout()
    plt.savefig(os.path.join("exploration_output", 'signs_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return sign_counts

def analyze_participants(df):
    """Анализ участников"""
    print("\n👥 Анализ участников...")
    
    participant_counts = df['participant_id'].value_counts()
    
    print(f"   Всего участников: {len(participant_counts)}")
    print(f"   Среднее количество записей на участника: {participant_counts.mean():.1f}")
    print(f"   Медиана записей на участника: {participant_counts.median():.1f}")
    print(f"   Минимум записей на участника: {participant_counts.min()}")
    print(f"   Максимум записей на участника: {participant_counts.max()}")
    
    # Визуализация
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(participant_counts.values, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Распределение количества записей на участника')
    plt.xlabel('Количество записей')
    plt.ylabel('Количество участников')
    
    plt.subplot(1, 2, 2)
    participant_counts.head(20).plot(kind='bar')
    plt.title('Топ-20 участников по количеству записей')
    plt.xlabel('ID участника')
    plt.ylabel('Количество записей')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join("exploration_output", 'participants_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return participant_counts

def analyze_landmark_files(df, sample_size=100):
    """Анализ файлов с landmarks"""
    print(f"\n🎯 Анализ landmark файлов (выборка из {sample_size} файлов)...")
    
    # Выбираем случайную выборку файлов для анализа
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    frame_counts = []
    landmark_counts = []
    file_sizes = []
    missing_data_ratios = []
    
    data_dir = Path(DIR_DATASET)
    
    for idx, row in sample_df.iterrows():
        file_path = data_dir / row['path']
        
        if file_path.exists():
            # Размер файла
            file_size = file_path.stat().st_size
            file_sizes.append(file_size)
            
            try:
                # Загружаем parquet файл
                landmarks_df = pd.read_parquet(file_path)
                
                # Анализируем структуру
                frame_counts.append(len(landmarks_df))
                
                # Предполагаем, что колонки содержат координаты landmarks
                # Формат может быть: frame, x_0, y_0, z_0, x_1, y_1, z_1, ...
                coord_columns = [col for col in landmarks_df.columns if col.startswith(('x_', 'y_', 'z_'))]
                if coord_columns:
                    landmark_count = len(coord_columns) // 3  # 3 координаты на landmark
                    landmark_counts.append(landmark_count)
                else:
                    # Альтернативный формат - все колонки кроме frame
                    non_frame_cols = [col for col in landmarks_df.columns if col != 'frame']
                    landmark_count = len(non_frame_cols) // 3
                    landmark_counts.append(landmark_count)
                
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
        print(f"     Количество точек: {landmark_counts[0]} (одинаково для всех)")
    
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
    
    # Визуализация
    if frame_counts:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(frame_counts, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Распределение количества кадров')
        plt.xlabel('Количество кадров')
        plt.ylabel('Частота')
        
        plt.subplot(1, 3, 2)
        plt.boxplot(frame_counts)
        plt.title('Box plot количества кадров')
        plt.ylabel('Количество кадров')
        
        plt.subplot(1, 3, 3)
        plt.hist(file_sizes, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Распределение размеров файлов')
        plt.xlabel('Размер файла (байты)')
        plt.ylabel('Частота')
        
        plt.tight_layout()
        plt.savefig(os.path.join("exploration_output", 'landmark_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    return {
        'frame_counts': frame_counts,
        'landmark_counts': landmark_counts,
        'file_sizes': file_sizes,
        'missing_data_ratios': missing_data_ratios
    }

def analyze_landmark_structure(sample_files=5):
    """Детальный анализ структуры landmark файлов"""
    print(f"\n🔍 Детальный анализ структуры landmark файлов (выборка из {sample_files} файлов)...")
    
    data_dir = Path(DIR_DATASET)
    train_df = pd.read_csv(os.path.join(DIR_DATASET, 'train.csv'))
    
    # Выбираем несколько файлов для детального анализа
    sample_df = train_df.sample(n=sample_files, random_state=42)
    
    for idx, row in sample_df.iterrows():
        file_path = data_dir / row['path']
        sign = row['sign']
        
        print(f"\n   Файл: {row['path']}")
        print(f"   Знак: {sign}")
        
        if file_path.exists():
            try:
                landmarks_df = pd.read_parquet(file_path)
                print(f"   Размер: {landmarks_df.shape}")
                print(f"   Колонки: {list(landmarks_df.columns)}")
                print(f"   Первые 3 строки:")
                print(landmarks_df.head(3))
                
                # Анализ типов данных
                print(f"   Типы данных:")
                print(landmarks_df.dtypes)
                
                # Проверка на NaN
                nan_count = landmarks_df.isnull().sum().sum()
                print(f"   NaN значений: {nan_count}")
                
            except Exception as e:
                print(f"   Ошибка при чтении: {e}")
        else:
            print(f"   Файл не найден")

def analyze_signs_by_participant(df):
    """Анализ распределения знаков по участникам"""
    print("\n👥📈 Анализ распределения знаков по участникам...")
    
    # Создаем матрицу участник-знак
    participant_sign_matrix = df.groupby(['participant_id', 'sign']).size().unstack(fill_value=0)
    
    print(f"   Размер матрицы: {participant_sign_matrix.shape}")
    print(f"   Участников: {participant_sign_matrix.shape[0]}")
    print(f"   Знаков: {participant_sign_matrix.shape[1]}")
    
    # Статистика по участникам
    signs_per_participant = participant_sign_matrix.sum(axis=1)
    print(f"   Среднее количество знаков на участника: {signs_per_participant.mean():.1f}")
    print(f"   Медиана знаков на участника: {signs_per_participant.median():.1f}")
    
    # Статистика по знакам
    participants_per_sign = (participant_sign_matrix > 0).sum(axis=0)
    print(f"   Среднее количество участников на знак: {participants_per_sign.mean():.1f}")
    print(f"   Медиана участников на знак: {participants_per_sign.median():.1f}")
    
    # Визуализация
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(signs_per_participant, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Распределение количества знаков на участника')
    plt.xlabel('Количество знаков')
    plt.ylabel('Количество участников')
    
    plt.subplot(2, 2, 2)
    plt.hist(participants_per_sign, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Распределение количества участников на знак')
    plt.xlabel('Количество участников')
    plt.ylabel('Количество знаков')
    
    plt.subplot(2, 2, 3)
    # Тепловая карта (топ-20 участников и топ-20 знаков)
    top_participants = signs_per_participant.nlargest(20).index
    top_signs = participants_per_sign.nlargest(20).index
    heatmap_data = participant_sign_matrix.loc[top_participants, top_signs]
    
    sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Количество записей'})
    plt.title('Тепловая карта: участники vs знаки (топ-20)')
    plt.xlabel('Знаки')
    plt.ylabel('Участники')
    
    plt.subplot(2, 2, 4)
    # Корреляция между участниками
    correlation_matrix = participant_sign_matrix.T.corr()
    sns.heatmap(correlation_matrix.iloc[:20, :20], cmap='coolwarm', center=0)
    plt.title('Корреляция между участниками (топ-20)')
    
    plt.tight_layout()
    plt.savefig(os.path.join("exploration_output", 'participant_sign_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

def generate_summary_report(df, sign_mapping, landmark_stats):
    """Генерация итогового отчета"""
    print("\n📋 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)
    
    print(f"📊 Общая статистика:")
    print(f"   Всего записей: {len(df):,}")
    print(f"   Уникальных знаков: {df['sign'].nunique()}")
    print(f"   Уникальных участников: {df['participant_id'].nunique()}")
    print(f"   Уникальных последовательностей: {df['sequence_id'].nunique()}")
    
    print(f"\n🎯 Статистика знаков:")
    sign_counts = df['sign'].value_counts()
    print(f"   Самый частый знак: {sign_counts.index[0]} ({sign_counts.iloc[0]} записей)")
    print(f"   Самый редкий знак: {sign_counts.index[-1]} ({sign_counts.iloc[-1]} записей)")
    print(f"   Среднее количество записей на знак: {sign_counts.mean():.1f}")
    
    print(f"\n👥 Статистика участников:")
    participant_counts = df['participant_id'].value_counts()
    print(f"   Самый активный участник: {participant_counts.index[0]} ({participant_counts.iloc[0]} записей)")
    print(f"   Среднее количество записей на участника: {participant_counts.mean():.1f}")
    
    if landmark_stats['frame_counts']:
        print(f"\n🎬 Статистика видео:")
        print(f"   Среднее количество кадров: {np.mean(landmark_stats['frame_counts']):.1f}")
        print(f"   Минимум кадров: {min(landmark_stats['frame_counts'])}")
        print(f"   Максимум кадров: {max(landmark_stats['frame_counts'])}")
    
    if landmark_stats['landmark_counts']:
        print(f"   Количество landmark точек: {landmark_stats['landmark_counts'][0]}")
    
    print(f"\n💾 Размер датасета:")
    total_size = sum(landmark_stats['file_sizes']) if landmark_stats['file_sizes'] else 0
    print(f"   Примерный размер landmark файлов: {total_size / (1024**3):.1f} GB")
    
    print("=" * 50)

def main():
    """Основная функция анализа"""
    print("🔍 АНАЛИЗ GOOGLE ASL SIGNS ДАТАСЕТА")
    print("=" * 50)
    
    # Загрузка данных
    sign_mapping = load_sign_mapping()
    df = load_train_data(sample_size=10000)  # Анализируем выборку для скорости
    
    # Анализы
    sign_counts = analyze_signs_distribution(df, sign_mapping)
    participant_counts = analyze_participants(df)
    landmark_stats = analyze_landmark_files(df, sample_size=200)
    analyze_signs_by_participant(df)
    
    # Детальный анализ структуры (небольшая выборка)
    analyze_landmark_structure(sample_files=3)
    
    # Итоговый отчет
    generate_summary_report(df, sign_mapping, landmark_stats)
    
    print("\n✅ Анализ данных завершен!")
    print("📁 Сохранены графики:")
    print("   - signs_distribution.png")
    print("   - participants_analysis.png") 
    print("   - landmark_analysis.png")
    print("   - participant_sign_analysis.png")

if __name__ == "__main__":
    main()