# 1. Анализ данных
python data_exploration.py

# 2. Тест препроцессинга  
python preprocessing.py

# 3. Тест аугментаций
python augmentations.py

# 4. Тест DataLoader
python data_loader.py

# 5. Полный тест пайплайна (рекомендуется)
python test_pipeline.py

# 6. Альтернативный способ - все вместе
python -c "
from data_loader import test_dataloader
from preprocessing import test_preprocessor  
from augmentations import test_augmentations

print('🚀 Полный тест пайплайна...')
test_preprocessor()
test_augmentations() 
test_dataloader()
print('✅ Готовы к обучению!')
"