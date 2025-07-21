import pandas as pd

# Загружаем файл датасета
df = pd.read_parquet('../dataset25/train_landmark_files/16069/10042041.parquet')

print("📊 Анализ структуры датасета:")
print(f"Общее количество записей: {len(df)}")
print(f"Уникальных landmarks: {len(df['landmark_index'].unique())}")
print(f"Типы landmarks: {df['type'].unique()}")

print("\nДетали по типам:")
for t in df['type'].unique():
    landmarks = sorted(df[df['type']==t]['landmark_index'].unique())
    print(f"  {t}: {len(landmarks)} landmarks ({min(landmarks)}-{max(landmarks)})")

print(f"\nКадры: {len(df['frame'].unique())} (диапазон: {df['frame'].min()}-{df['frame'].max()})")

# Проверяем структуру тензора
frames = sorted(df['frame'].unique())
all_landmarks = sorted(df['landmark_index'].unique())

print(f"\nСтруктура тензора:")
print(f"  Кадры: {len(frames)}")
print(f"  Landmarks: {len(all_landmarks)}")
print(f"  Размер тензора: ({len(frames)}, {len(all_landmarks)}, 3)") 