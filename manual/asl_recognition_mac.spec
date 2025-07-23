# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec файл для ASL Recognition macOS приложения

Использование:
1. Установите PyInstaller: pip install pyinstaller
2. Создайте приложение: pyinstaller asl_recognition_mac.spec
3. Приложение будет в папке dist/

Для создания .dmg файла:
4. Установите create-dmg: brew install create-dmg
5. Создайте DMG: create-dmg --volname "ASL Recognition" --window-pos 200 120 --window-size 600 300 --icon-size 100 --icon "ASL Recognition.app" 175 120 --hide-extension "ASL Recognition.app" --app-drop-link 425 120 "ASL Recognition.dmg" "dist/"
"""

import os
import sys
from pathlib import Path

# Определяем пути
block_cipher = None
current_dir = Path.cwd()

# Находим все необходимые данные
data_files = []

# Добавляем модели (если есть)
models_dir = current_dir / 'models'
if models_dir.exists():
    for model_file in models_dir.glob('*.pth'):
        data_files.append((str(model_file), 'models'))
    for manifest_file in models_dir.glob('*.json'):
        data_files.append((str(manifest_file), 'models'))

# Добавляем маппинги знаков
dataset_dir = current_dir / 'dataset25'
if dataset_dir.exists():
    for dataset_file in dataset_dir.glob('*.json'):
        data_files.append((str(dataset_file), 'dataset25'))

# Альтернативные пути для данных
manual_models = current_dir / 'manual' / 'models'
if manual_models.exists():
    for model_file in manual_models.glob('*.pth'):
        data_files.append((str(model_file), 'models'))
    for manifest_file in manual_models.glob('*.json'):
        data_files.append((str(manifest_file), 'models'))

manual_dataset = current_dir / 'manual' / 'dataset25'
if manual_dataset.exists():
    for dataset_file in manual_dataset.glob('*.json'):
        data_files.append((str(dataset_file), 'dataset25'))

print(f"Найдены файлы данных: {data_files}")

# Дополнительные модули для MediaPipe и OpenCV
hidden_imports = [
    'cv2',
    'mediapipe',
    'numpy',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.nn.init',
    'torch.backends.mps',
    'torchvision',
    'PIL',
    'PIL.Image',
    'google.protobuf',
    'google.protobuf.pyext',
    'google.protobuf.pyext._message',
    'step2_prepare_dataset',
    'step3_prepare_train',
    'pathlib',
    'typing',
    'datetime',
    'platform',
    'json',
    'time',
    'math',
    'os',
    'sys'
]

# Анализируем основной скрипт
a = Analysis(
    ['step5_live_recognition_mac.py'],
    pathex=[str(current_dir)],
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'IPython',
        'notebook',
        'pytest',
        'unittest',
        'test',
        'distutils',
        'setuptools',
        'pip',
        'wheel'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Удаляем ненужные файлы для уменьшения размера
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Создаем исполняемый файл
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ASL Recognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,  # Запускаем без консоли
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Создаем коллекцию файлов
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='ASL Recognition'
)

# Создаем macOS приложение (.app bundle)
app = BUNDLE(
    coll,
    name='ASL Recognition.app',
    icon=None,  # Можно добавить иконку
    bundle_identifier='com.asl.recognition',
    version='1.0.0',
    info_plist={
        'CFBundleName': 'ASL Recognition',
        'CFBundleDisplayName': 'ASL Recognition',
        'CFBundleGetInfoString': 'ASL Recognition 1.0.0',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSCameraUsageDescription': 'This app needs camera access for ASL gesture recognition',
        'LSMinimumSystemVersion': '10.15.0',
        'NSRequiresAquaSystemAppearance': False,
        'LSApplicationCategoryType': 'public.app-category.education',
        'CFBundleDocumentTypes': [],
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'LSBackgroundOnly': False,
        'LSUIElement': False
    }
) 