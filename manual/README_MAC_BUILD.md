# 🍎 Создание macOS приложения ASL Recognition

Это руководство поможет вам создать standalone macOS приложение (.app) из Python кода для распознавания ASL жестов.

## 📋 Требования

- **macOS 10.15+** (Catalina или новее)
- **Python 3.8+**
- **Homebrew** (рекомендуется)
- **Веб-камера**
- **8+ GB RAM** (для PyTorch)

## 🚀 Быстрый старт

### Вариант 1: py2app (Рекомендуется для macOS)

```bash
# 1. Установите зависимости
pip install py2app

# 2. Перейдите в папку manual
cd manual

# 3. Создайте приложение
python setup_mac.py py2app

# 4. Приложение будет в папке dist/ASL Recognition.app
```

### Вариант 2: PyInstaller (Универсальный)

```bash
# 1. Установите PyInstaller
pip install pyinstaller

# 2. Перейдите в папку manual
cd manual

# 3. Создайте приложение
pyinstaller asl_recognition_mac.spec

# 4. Приложение будет в папке dist/ASL Recognition.app
```

## 📦 Подробные инструкции

### Шаг 1: Подготовка окружения

1. **Обновите macOS до последней версии**

2. **Установите Homebrew** (если не установлен):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Установите Python** (если не установлен):
   ```bash
   brew install python@3.11
   ```

4. **Создайте виртуальное окружение**:
   ```bash
   python3 -m venv asl_env
   source asl_env/bin/activate
   ```

### Шаг 2: Установка зависимостей

```bash
# Основные зависимости
pip install torch torchvision torchaudio
pip install opencv-python
pip install mediapipe
pip install numpy
pip install Pillow

# Инструменты для сборки
pip install py2app pyinstaller

# Дополнительные зависимости (если нужны)
pip install -r requirements.txt
```

### Шаг 3: Проверка моделей

Убедитесь, что у вас есть обученные модели:

```bash
# Проверьте наличие моделей
ls -la models/
ls -la manual/models/

# Должны быть файлы .pth и .json
```

### Шаг 4: Сборка с py2app

```bash
cd manual

# Тестовая сборка (быстрая, для проверки)
python setup_mac.py py2app -A

# Полная сборка (standalone)
python setup_mac.py py2app

# Очистка (если нужно пересобрать)
python setup_mac.py clean
```

### Шаг 5: Сборка с PyInstaller

```bash
cd manual

# Создание приложения
pyinstaller asl_recognition_mac.spec

# Если нужно пересобрать
rm -rf build dist
pyinstaller asl_recognition_mac.spec
```

## 🎯 Выбор версии приложения

У вас есть 3 версии на выбор:

### 1. Консольная версия
- **Файл**: `step5_live_recognition_mac.py`
- **Запуск**: `python step5_live_recognition_mac.py`
- **Плюсы**: Простая, быстрая
- **Минусы**: Консольный интерфейс

### 2. GUI версия
- **Файл**: `asl_recognition_gui_mac.py`
- **Запуск**: `python asl_recognition_gui_mac.py`
- **Плюсы**: Удобный интерфейс, настройки
- **Минусы**: Чуть больше размер

### 3. Оригинальная версия (адаптированная)
- **Файл**: `step5_live_recognition.py`
- **Запуск**: Требует CUDA (не для Mac)

## 🛠 Настройка сборки

### Редактирование setup_mac.py

Для настройки сборки py2app отредактируйте `setup_mac.py`:

```python
# Изменить название приложения
APP_NAME = 'Ваше название'

# Добавить иконку
'iconfile': 'path/to/icon.icns',

# Изменить версию
VERSION = '2.0.0'
```

### Редактирование asl_recognition_mac.spec

Для PyInstaller отредактируйте `asl_recognition_mac.spec`:

```python
# Изменить название
name='Ваше название.app'

# Добавить иконку
icon='path/to/icon.ico'

# Изменить bundle identifier
bundle_identifier='com.yourcompany.yourapp'
```

## 📱 Создание иконки

Создайте иконку в формате .icns:

```bash
# Установите инструменты
brew install imagemagick

# Создайте иконку из PNG
mkdir icon.iconset
sips -z 16 16 your_icon.png --out icon.iconset/icon_16x16.png
sips -z 32 32 your_icon.png --out icon.iconset/icon_16x16@2x.png
sips -z 32 32 your_icon.png --out icon.iconset/icon_32x32.png
sips -z 64 64 your_icon.png --out icon.iconset/icon_32x32@2x.png
sips -z 128 128 your_icon.png --out icon.iconset/icon_128x128.png
sips -z 256 256 your_icon.png --out icon.iconset/icon_128x128@2x.png
sips -z 256 256 your_icon.png --out icon.iconset/icon_256x256.png
sips -z 512 512 your_icon.png --out icon.iconset/icon_256x256@2x.png
sips -z 512 512 your_icon.png --out icon.iconset/icon_512x512.png
sips -z 1024 1024 your_icon.png --out icon.iconset/icon_512x512@2x.png

# Создайте .icns файл
iconutil -c icns icon.iconset
```

## 🗜 Создание DMG инсталлятора

После сборки приложения создайте DMG:

```bash
# Установите create-dmg
brew install create-dmg

# Создайте DMG
create-dmg \
  --volname "ASL Recognition" \
  --volicon "app_icon.icns" \
  --window-pos 200 120 \
  --window-size 600 300 \
  --icon-size 100 \
  --icon "ASL Recognition.app" 175 120 \
  --hide-extension "ASL Recognition.app" \
  --app-drop-link 425 120 \
  "ASL Recognition.dmg" \
  "dist/"
```

## 🚨 Решение проблем

### Проблема: "App не открывается"

```bash
# Снимите карантин (если скачано из интернета)
xattr -dr com.apple.quarantine "ASL Recognition.app"

# Разрешите в Настройки > Безопасность
```

### Проблема: "Модуль не найден"

```bash
# Проверьте, что все зависимости установлены
pip list | grep opencv
pip list | grep mediapipe
pip list | grep torch

# Переустановите проблемные модули
pip uninstall opencv-python
pip install opencv-python
```

### Проблема: "Камера не работает"

1. Разрешите доступ к камере в **Настройки > Безопасность > Конфиденциальность > Камера**
2. Проверьте ID камеры в приложении (обычно 0 или 1)

### Проблема: "Большой размер приложения"

```bash
# Исключите ненужные модули в setup_mac.py
EXCLUDES = [
    'matplotlib',  # ~100MB
    'scipy',       # ~50MB
    'pandas',      # ~30MB
    'jupyter',     # ~20MB
    # добавьте другие ненужные модули
]

# Используйте UPX для сжатия (в PyInstaller)
upx=True
```

## 📊 Размеры приложений

| Метод | Размер | Время сборки | Совместимость |
|-------|--------|--------------|---------------|
| py2app | ~800MB | 5-10 мин | Отличная |
| PyInstaller | ~1.2GB | 3-8 мин | Хорошая |
| py2app -A | ~50MB | 1 мин | Только разработка |

## 🎯 Рекомендации

### Для распространения:
- Используйте **py2app** (лучше для Mac)
- Создайте **DMG инсталлятор**
- Добавьте **цифровую подпись** (для App Store)

### Для разработки:
- Используйте **py2app -A** (режим алиаса)
- Быстрая пересборка
- Простое тестирование

### Для кроссплатформенности:
- Используйте **PyInstaller**
- Поддержка Windows/Linux
- Больше настроек

## 🔗 Полезные ссылки

- [py2app Documentation](https://py2app.readthedocs.io/)
- [PyInstaller Manual](https://pyinstaller.readthedocs.io/)
- [Apple Developer Guidelines](https://developer.apple.com/macos/human-interface-guidelines/)
- [macOS App Store Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)

## 💡 Советы

1. **Тестируйте на чистой системе** перед распространением
2. **Используйте виртуальное окружение** для изолированной сборки
3. **Проверяйте права доступа** к камере и файлам
4. **Документируйте системные требования** в README
5. **Создавайте резервные копии** рабочих моделей

## 📝 Лицензирование

При распространении приложения учтите лицензии:
- **PyTorch**: BSD-style license
- **OpenCV**: Apache 2.0
- **MediaPipe**: Apache 2.0
- **Ваш код**: Укажите свою лицензию

---

🎉 **Поздравляем!** Теперь у вас есть полнофункциональное macOS приложение для распознавания ASL жестов! 