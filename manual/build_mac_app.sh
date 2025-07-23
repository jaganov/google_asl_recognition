#!/bin/bash

# 🍎 Автоматическая сборка ASL Recognition для macOS
# Автор: ASL Recognition Team
# Версия: 1.0

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[BUILD]${NC} $1"
}

# Проверка macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    error "Этот скрипт предназначен только для macOS"
    exit 1
fi

log "🍎 Начинаем сборку ASL Recognition для macOS"

# Переменные
APP_NAME="ASL Recognition"
BUILD_METHOD=""
GUI_MODE=false
CLEAN_BUILD=false

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            BUILD_METHOD="$2"
            shift 2
            ;;
        --gui)
            GUI_MODE=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help|-h)
            echo "Использование: $0 [опции]"
            echo ""
            echo "Опции:"
            echo "  --method py2app|pyinstaller  Метод сборки (по умолчанию: py2app)"
            echo "  --gui                        Использовать GUI версию"
            echo "  --clean                      Очистить перед сборкой"
            echo "  --help, -h                   Показать эту справку"
            echo ""
            echo "Примеры:"
            echo "  $0                           # Сборка с py2app (консольная версия)"
            echo "  $0 --gui                     # Сборка GUI версии"
            echo "  $0 --method pyinstaller      # Сборка с PyInstaller"
            echo "  $0 --clean --gui             # Очистка и сборка GUI"
            exit 0
            ;;
        *)
            error "Неизвестная опция: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

# Установка метода по умолчанию
if [[ -z "$BUILD_METHOD" ]]; then
    BUILD_METHOD="py2app"
fi

# Проверка метода сборки
if [[ "$BUILD_METHOD" != "py2app" && "$BUILD_METHOD" != "pyinstaller" ]]; then
    error "Неподдерживаемый метод сборки: $BUILD_METHOD"
    exit 1
fi

log "Метод сборки: $BUILD_METHOD"
if $GUI_MODE; then
    log "Режим: GUI версия"
else
    log "Режим: Консольная версия"
fi

# Переходим в папку manual
if [[ ! -d "manual" ]]; then
    error "Папка 'manual' не найдена. Запустите скрипт из корня проекта."
    exit 1
fi

cd manual
log "Переход в папку manual: $(pwd)"

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    error "Python 3 не найден. Установите Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
log "Найден $PYTHON_VERSION"

# Проверка виртуального окружения
if [[ -z "$VIRTUAL_ENV" ]]; then
    warn "Виртуальное окружение не активировано"
    read -p "Продолжить без виртуального окружения? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Активируйте виртуальное окружение и запустите скрипт снова"
        exit 1
    fi
else
    log "Используется виртуальное окружение: $VIRTUAL_ENV"
fi

# Проверка зависимостей
info "Проверка зависимостей..."

check_package() {
    if python3 -c "import $1" &> /dev/null; then
        log "✅ $1 установлен"
    else
        error "❌ $1 не установлен"
        echo "Установите: pip install $2"
        exit 1
    fi
}

check_package "cv2" "opencv-python"
check_package "mediapipe" "mediapipe"
check_package "torch" "torch"
check_package "numpy" "numpy"

if [[ "$BUILD_METHOD" == "py2app" ]]; then
    check_package "py2app" "py2app"
else
    check_package "PyInstaller" "pyinstaller"
fi

# Проверка моделей
info "Проверка моделей..."
MODEL_FOUND=false

check_models_in() {
    local dir=$1
    if [[ -d "$dir" ]]; then
        local model_count=$(find "$dir" -name "*.pth" | wc -l)
        if [[ $model_count -gt 0 ]]; then
            log "✅ Найдено $model_count моделей в $dir"
            MODEL_FOUND=true
        fi
    fi
}

check_models_in "models"
check_models_in "../models"

if ! $MODEL_FOUND; then
    warn "⚠️ Модели не найдены. Приложение может не работать без моделей."
    read -p "Продолжить сборку? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Выбор главного файла
if $GUI_MODE; then
    MAIN_FILE="asl_recognition_gui_mac.py"
    info "Главный файл: $MAIN_FILE (GUI версия)"
else
    MAIN_FILE="step5_live_recognition_mac.py"
    info "Главный файл: $MAIN_FILE (консольная версия)"
fi

# Проверка главного файла
if [[ ! -f "$MAIN_FILE" ]]; then
    error "Главный файл не найден: $MAIN_FILE"
    exit 1
fi

# Очистка (если требуется)
if $CLEAN_BUILD; then
    info "Очистка предыдущих сборок..."
    rm -rf build dist
    log "✅ Очистка завершена"
fi

# Создание папки для результатов
mkdir -p dist

# Сборка приложения
info "🔨 Начинаем сборку с $BUILD_METHOD..."

if [[ "$BUILD_METHOD" == "py2app" ]]; then
    # Сборка с py2app
    if $GUI_MODE; then
        # Для GUI нужно модифицировать setup_mac.py
        info "Настройка setup_mac.py для GUI версии..."
        sed -i.bak "s/step5_live_recognition_mac.py/asl_recognition_gui_mac.py/" setup_mac.py
    fi
    
    log "Запуск py2app..."
    python3 setup_mac.py py2app
    
    # Восстанавливаем оригинальный setup_mac.py
    if $GUI_MODE && [[ -f "setup_mac.py.bak" ]]; then
        mv setup_mac.py.bak setup_mac.py
    fi
    
    APP_PATH="dist/$APP_NAME.app"
    
else
    # Сборка с PyInstaller
    if $GUI_MODE; then
        # Создаем временный spec файл для GUI
        info "Создание spec файла для GUI версии..."
        SPEC_FILE="asl_recognition_gui_mac.spec"
        sed "s/step5_live_recognition_mac.py/asl_recognition_gui_mac.py/" asl_recognition_mac.spec > "$SPEC_FILE"
    else
        SPEC_FILE="asl_recognition_mac.spec"
    fi
    
    log "Запуск PyInstaller..."
    pyinstaller "$SPEC_FILE"
    
    # Удаляем временный spec файл
    if $GUI_MODE && [[ -f "asl_recognition_gui_mac.spec" ]]; then
        rm "asl_recognition_gui_mac.spec"
    fi
    
    APP_PATH="dist/$APP_NAME.app"
fi

# Проверка результата
if [[ -d "$APP_PATH" ]]; then
    log "✅ Сборка завершена успешно!"
    
    # Информация о размере
    APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)
    log "📦 Размер приложения: $APP_SIZE"
    
    # Проверка структуры приложения
    if [[ -f "$APP_PATH/Contents/MacOS/$APP_NAME" ]]; then
        log "✅ Исполняемый файл найден"
    else
        warn "⚠️ Исполняемый файл не найден"
    fi
    
    # Информация о местоположении
    ABS_PATH=$(cd "$(dirname "$APP_PATH")" && pwd)/$(basename "$APP_PATH")
    info "📍 Путь к приложению: $ABS_PATH"
    
    # Снятие карантина (для безопасности)
    log "Снятие карантина с приложения..."
    xattr -dr com.apple.quarantine "$APP_PATH" 2>/dev/null || true
    
    # Предложение тестирования
    echo ""
    log "🎉 Приложение готово к использованию!"
    echo ""
    echo "Следующие шаги:"
    echo "1. Протестируйте приложение: open '$ABS_PATH'"
    echo "2. Разрешите доступ к камере в Настройках системы"
    echo "3. Создайте DMG инсталлятор (опционально):"
    echo "   brew install create-dmg"
    echo "   create-dmg --volname '$APP_NAME' '$APP_NAME.dmg' 'dist/'"
    echo ""
    
    # Предложение автоматического запуска
    read -p "Открыть приложение сейчас? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Запускаем приложение..."
        open "$APP_PATH"
    fi
    
else
    error "❌ Сборка не удалась. Приложение не найдено в $APP_PATH"
    echo ""
    echo "Проверьте сообщения об ошибках выше и попробуйте:"
    echo "1. Переустановить зависимости"
    echo "2. Использовать другой метод сборки"
    echo "3. Запустить с флагом --clean"
    exit 1
fi

log "�� Сборка завершена!" 