#!/usr/bin/env python3
"""
Тестовый скрипт для проверки всего пайплайна
"""

def test_imports():
    """Тест импортов"""
    print("🔍 Проверяем импорты...")
    
    try:
        from preprocessing import ASLPreprocessor
        print("   ✅ ASLPreprocessor импортирован")
    except Exception as e:
        print(f"   ❌ Ошибка импорта ASLPreprocessor: {e}")
        return False
    
    try:
        from augmentations import ASLAugmentations
        print("   ✅ ASLAugmentations импортирован")
    except Exception as e:
        print(f"   ❌ Ошибка импорта ASLAugmentations: {e}")
        return False
    
    try:
        from data_loader import ASLDataLoader
        print("   ✅ ASLDataLoader импортирован")
    except Exception as e:
        print(f"   ❌ Ошибка импорта ASLDataLoader: {e}")
        return False
    
    return True

def test_preprocessor():
    """Тест препроцессора"""
    print("\n🧪 Тестируем препроцессор...")
    
    try:
        from preprocessing import test_preprocessor
        success = test_preprocessor()
        if success:
            print("   ✅ Препроцессор работает")
            return True
        else:
            print("   ❌ Проблемы с препроцессором")
            return False
    except Exception as e:
        print(f"   ❌ Ошибка теста препроцессора: {e}")
        return False

def test_augmentations():
    """Тест аугментаций"""
    print("\n🧪 Тестируем аугментации...")
    
    try:
        from augmentations import test_augmentations
        test_augmentations()
        print("   ✅ Аугментации работают")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка теста аугментаций: {e}")
        return False

def test_dataloader():
    """Тест загрузчика данных"""
    print("\n🧪 Тестируем загрузчик данных...")
    
    try:
        from data_loader import test_dataloader
        success = test_dataloader()
        if success:
            print("   ✅ Загрузчик данных работает")
            return True
        else:
            print("   ❌ Проблемы с загрузчиком данных")
            return False
    except Exception as e:
        print(f"   ❌ Ошибка теста загрузчика данных: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Запускаем полный тест пайплайна...\n")
    
    # Тест импортов
    if not test_imports():
        print("\n❌ Проблемы с импортами. Остановка.")
        return
    
    # Тест компонентов
    tests = [
        test_preprocessor,
        test_augmentations,
        test_dataloader
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Итоговый результат
    print("\n" + "="*50)
    if all(results):
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Пайплайн готов к работе!")
    else:
        print("⚠️ Некоторые тесты не прошли. Проверьте ошибки выше.")
    print("="*50)

if __name__ == "__main__":
    main() 