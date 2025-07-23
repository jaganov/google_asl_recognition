"""
Setup script для создания macOS приложения ASL Recognition
Использует py2app для упаковки Python кода в .app файл

Установка зависимостей:
pip install py2app

Создание приложения:
python setup_mac.py py2app

Создание в режиме алиаса (для разработки):
python setup_mac.py py2app -A
"""

from setuptools import setup
import py2app
import os
import sys
from pathlib import Path

APP = ['step5_live_recognition_mac.py']
APP_NAME = 'ASL Recognition'
VERSION = '1.0.0'

# Находим все необходимые файлы данных
def find_data_files():
    data_files = []
    
    # Модели (если есть)
    models_dir = Path('models')
    if models_dir.exists():
        model_files = []
        for model_file in models_dir.glob('*.pth'):
            model_files.append(str(model_file))
        if model_files:
            data_files.append(('models', model_files))
    
    # Маппинги знаков
    dataset_dir = Path('dataset25')
    if dataset_dir.exists():
        dataset_files = []
        for dataset_file in dataset_dir.glob('*.json'):
            dataset_files.append(str(dataset_file))
        if dataset_files:
            data_files.append(('dataset25', dataset_files))
    
    return data_files

# Получаем пути для включения
DATA_FILES = find_data_files()

# Дополнительные модули, которые нужно включить
INCLUDES = [
    'cv2',
    'mediapipe',
    'numpy',
    'torch',
    'json',
    'pathlib',
    'typing',
    'datetime',
    'platform',
    'step2_prepare_dataset',
    'step3_prepare_train'
]

# Пакеты, которые нужно включить полностью
PACKAGES = [
    'mediapipe',
    'cv2',
    'torch',
    'torchvision',
    'numpy',
    'PIL'
]

# Модули, которые нужно исключить (для уменьшения размера)
EXCLUDES = [
    'tkinter',
    'unittest',
    'test',
    'distutils',
    'setuptools',
    'email',
    'html',
    'http',
    'urllib',
    'xml',
    'pydoc_data',
    'bz2',
    'lzma',
    'zipfile',
    'tarfile',
    'gzip',
    'ftplib',
    'subprocess',
    'multiprocessing',
    'concurrent',
    'socket',
    'select',
    'ssl',
    'hashlib',
    'hmac',
    'base64',
    'quopri',
    'uu',
    'binascii'
]

# Ресурсы для включения
RESOURCES = []

# Настройки приложения
OPTIONS = {
    'argv_emulation': False,
    'strip': True,
    'compressed': True,
    'optimize': 2,
    'includes': INCLUDES,
    'packages': PACKAGES,
    'excludes': EXCLUDES,
    'resources': RESOURCES,
    'iconfile': None,  # Можно добавить иконку если есть
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': APP_NAME,
        'CFBundleGetInfoString': f'{APP_NAME} {VERSION}',
        'CFBundleVersion': VERSION,
        'CFBundleShortVersionString': VERSION,
        'NSHighResolutionCapable': True,
        'NSCameraUsageDescription': 'This app needs camera access for ASL gesture recognition',
        'LSMinimumSystemVersion': '10.15.0',
        'CFBundleIdentifier': 'com.asl.recognition',
        'NSRequiresAquaSystemAppearance': False,
        'LSApplicationCategoryType': 'public.app-category.education',
        'CFBundleDocumentTypes': [],
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False
    }
}

setup(
    app=APP,
    name=APP_NAME,
    version=VERSION,
    description='ASL Gesture Recognition for macOS',
    author='ASL Recognition Team',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    python_requires='>=3.8',
)

print(f"""
🍎 Setup script для {APP_NAME} готов!

Для создания приложения выполните:

1. Установите py2app:
   pip install py2app

2. Создайте приложение:
   python setup_mac.py py2app

3. Приложение будет создано в папке dist/

Дополнительные команды:
- Режим разработки (алиас): python setup_mac.py py2app -A
- Очистка: python setup_mac.py clean

Требования:
- macOS 10.15+ (Catalina)
- Доступ к камере
- Модели должны находиться в папке models/
""") 