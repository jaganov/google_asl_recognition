import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from step2_prepare_dataset import load_dataset

class MediaPipeCapture:
    """
    Класс для захвата жестов с веб-камеры через MediaPipe
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 max_frames: int = 100,
                 fps: int = 30,
                 show_video: bool = True,
                 target_frames: int = 16,  # Целевое количество кадров как в датасете
                 use_only_face: bool = True):  # Использовать только face landmarks (как в датасете)
        """
        Args:
            camera_id: ID камеры (обычно 0 для встроенной)
            max_frames: Максимальное количество кадров для записи
            fps: Частота кадров в секунду
            show_video: Показывать ли видео во время записи
            target_frames: Целевое количество кадров (как в датасете)
            use_only_face: Использовать только face landmarks (как в датасете)
        """
        self.camera_id = camera_id
        self.max_frames = max_frames
        self.fps = fps
        self.show_video = show_video
        self.target_frames = target_frames
        self.use_only_face = use_only_face
        
        # Инициализация MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Настройки MediaPipe
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Хранение данных
        self.landmarks_data = []
        self.frame_timestamps = []
        
        print(f"🎥 MediaPipe Capture инициализирован:")
        print(f"   Камера: {camera_id}")
        print(f"   Максимум кадров: {max_frames}")
        print(f"   Целевых кадров: {target_frames}")
        print(f"   FPS: {fps}")
        print(f"   Показ видео: {show_video}")
        print(f"   Только face landmarks: {use_only_face} (как в датасете)")
    
    def start_capture(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Запускает захват жестов с веб-камеры
        
        Returns:
            landmarks_data: Список кадров с landmarks
            frame_timestamps: Временные метки кадров
        """
        print("\n🎬 Начинаем захват жестов...")
        print("   Нажмите 'q' для остановки записи")
        print("   Нажмите 's' для сохранения текущего жеста")
        
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("❌ Ошибка: Не удалось открыть камеру")
            return [], []
        
        # Настройка камеры
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.landmarks_data = []
        self.frame_timestamps = []
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while frame_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Ошибка чтения кадра")
                    break
                
                # Конвертируем BGR в RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Обрабатываем кадр через MediaPipe
                results = self.holistic.process(rgb_frame)
                
                # Извлекаем landmarks
                frame_landmarks = self._extract_landmarks(results)
                
                if frame_landmarks is not None:
                    self.landmarks_data.append(frame_landmarks)
                    self.frame_timestamps.append(time.time() - start_time)
                    frame_count += 1
                
                # Рисуем landmarks на кадре
                annotated_frame = self._draw_landmarks(frame, results)
                
                # Показываем информацию на кадре
                cv2.putText(annotated_frame, f"Frames: {frame_count}/{self.max_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Press 'q' to stop, 's' to save", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if self.show_video:
                    cv2.imshow('MediaPipe Capture', annotated_frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("   Остановка записи...")
                    break
                elif key == ord('s'):
                    print("   Сохранение жеста...")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"✅ Захват завершен: {len(self.landmarks_data)} кадров")
        return self.landmarks_data, self.frame_timestamps
    
    def _extract_landmarks(self, results) -> Optional[np.ndarray]:
        """
        Извлекает landmarks из результатов MediaPipe
        
        Args:
            results: Результаты обработки MediaPipe
            
        Returns:
            landmarks: Массив landmarks (N, 3) или None
        """
        landmarks = []
        
        if self.use_only_face:
            # Используем только face landmarks (как в датасете)
            if results.face_landmarks:
                for landmark in results.face_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            # Используем все landmarks (face + pose + hands)
            # Face landmarks (468 точек)
            if results.face_landmarks:
                for landmark in results.face_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Pose landmarks (33 точки)
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Left hand landmarks (21 точка)
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Right hand landmarks (21 точка)
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        if landmarks:
            return np.array(landmarks).reshape(-1, 3)
        return None
    
    def _draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Рисует landmarks на кадре
        
        Args:
            frame: Исходный кадр
            results: Результаты MediaPipe
            
        Returns:
            annotated_frame: Кадр с нарисованными landmarks
        """
        annotated_frame = frame.copy()
        
        # Рисуем face landmarks
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Рисуем pose landmarks (только если не используем только face)
        if not self.use_only_face and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Рисуем left hand landmarks (только если не используем только face)
        if not self.use_only_face and results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Рисуем right hand landmarks (только если не используем только face)
        if not self.use_only_face and results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return annotated_frame
    
    def normalize_landmarks(self, landmarks_data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Нормализует landmarks к формату датасета
        
        Args:
            landmarks_data: Список кадров с landmarks
            
        Returns:
            normalized_data: Нормализованные данные
        """
        if not landmarks_data:
            return []
        
        print(f"🔄 Нормализация landmarks:")
        print(f"   Исходных кадров: {len(landmarks_data)}")
        print(f"   Исходных landmarks на кадр: {len(landmarks_data[0]) if landmarks_data else 0}")
        
        # 1. Ограничиваем количество кадров
        if len(landmarks_data) > self.target_frames:
            # Выбираем равномерно распределенные кадры
            indices = np.linspace(0, len(landmarks_data) - 1, self.target_frames, dtype=int)
            landmarks_data = [landmarks_data[i] for i in indices]
            print(f"   Ограничено до {self.target_frames} кадров")
        
        # 2. Проверяем количество landmarks
        if self.use_only_face:
            expected_landmarks = 468  # Face landmarks (как в старом формате датасета)
            print(f"   Используем только face landmarks: {expected_landmarks} точек")
        else:
            # Face (468) + Pose (33) + Left Hand (21) + Right Hand (21) = 543
            expected_landmarks = 543  # Полный набор landmarks (как в исправленном датасете)
            print(f"   Используем все landmarks: {expected_landmarks} точек")
            print(f"     - Face: 468 (0-467)")
            print(f"     - Pose: 33 (468-500)") 
            print(f"     - Left Hand: 21 (501-521)")
            print(f"     - Right Hand: 21 (522-542)")
        
        # 3. Приводим к правильному количеству landmarks
        normalized_data = []
        for frame in landmarks_data:
            if len(frame) >= expected_landmarks:
                # Берем первые expected_landmarks точек
                normalized_frame = frame[:expected_landmarks]
            else:
                # Дополняем нулями если недостаточно точек
                normalized_frame = np.zeros((expected_landmarks, 3))
                normalized_frame[:len(frame)] = frame
            
            normalized_data.append(normalized_frame)
        
        print(f"   Нормализованных кадров: {len(normalized_data)}")
        print(f"   Нормализованных landmarks на кадр: {len(normalized_data[0])}")
        
        return normalized_data
    
    def convert_to_dataset_format(self) -> torch.Tensor:
        """
        Конвертирует захваченные данные в формат датасета
        
        Returns:
            tensor: Тензор размером (frames, landmarks, 3) в формате датасета
        """
        if not self.landmarks_data:
            return torch.empty(0, 0, 3)
        
        # Нормализуем данные
        normalized_data = self.normalize_landmarks(self.landmarks_data)
        
        if not normalized_data:
            return torch.empty(0, 0, 3)
        
        # Конвертируем в тензор
        tensor = torch.tensor(normalized_data, dtype=torch.float32)
        
        print(f"✅ Конвертировано в формат датасета: {tensor.shape}")
        return tensor
    
    def save_gesture(self, filename: str = "captured_gesture.json") -> None:
        """
        Сохраняет захваченный жест в JSON файл
        
        Args:
            filename: Имя файла для сохранения
        """
        if not self.landmarks_data:
            print("❌ Нет данных для сохранения")
            return
        
        # Нормализуем данные
        normalized_data = self.normalize_landmarks(self.landmarks_data)
        
        # Подготавливаем данные для сохранения
        gesture_data = {
            'frames': len(normalized_data),
            'landmarks_per_frame': len(normalized_data[0]) if normalized_data else 0,
            'timestamps': self.frame_timestamps[:len(normalized_data)],
            'landmarks': [frame.tolist() for frame in normalized_data],
            'dataset_format': True,
            'target_frames': self.target_frames,
            'use_only_face': self.use_only_face
        }
        
        # Сохраняем в JSON
        with open(filename, 'w') as f:
            json.dump(gesture_data, f, indent=2)
        
        print(f"✅ Жест сохранен в {filename}")
        print(f"   Кадров: {len(normalized_data)}")
        print(f"   Landmarks на кадр: {len(normalized_data[0]) if normalized_data else 0}")
        print(f"   Формат датасета: {gesture_data['dataset_format']}")
    
    def convert_to_tensor(self) -> torch.Tensor:
        """
        Конвертирует захваченные данные в тензор (устаревший метод)
        
        Returns:
            tensor: Тензор размером (frames, landmarks, 3)
        """
        return self.convert_to_dataset_format()

def visualize_comparison(captured_tensor: torch.Tensor, 
                        dataset_tensor: torch.Tensor,
                        captured_title: str = "Захваченный жест",
                        dataset_title: str = "Жест из датасета",
                        figsize: Tuple[int, int] = (20, 8)) -> None:
    """
    Визуализирует сравнение захваченного жеста с жестом из датасета
    """
    fig = plt.figure(figsize=figsize)
    
    # График захваченного жеста
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    captured_data = captured_tensor.numpy()
    
    if len(captured_data) > 0:
        # Убираем NaN значения
        valid_mask = ~np.isnan(captured_data).any(axis=2)
        valid_frames = []
        for frame in captured_data:
            valid_frame = frame[~np.isnan(frame).any(axis=1)]
            if len(valid_frame) > 0:
                valid_frames.append(valid_frame)
        
        if valid_frames:
            all_points = np.vstack(valid_frames)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
            
            # Добавляем отступы
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            z_margin = (z_max - z_min) * 0.1
            
            ax1.set_xlim(x_min - x_margin, x_max + x_margin)
            ax1.set_ylim(y_min - y_margin, y_max + y_margin)
            ax1.set_zlim(z_min - z_margin, z_max + z_margin)
            
            # Рисуем точки
            ax1.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], 
                       c='blue', s=20, alpha=0.7, label=f'Landmarks ({len(all_points)} точек)')
    
    ax1.set_xlabel('X координата')
    ax1.set_ylabel('Y координата')
    ax1.set_zlabel('Z координата')
    ax1.set_title(f"{captured_title}\nКадров: {len(captured_data)}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_box_aspect([1, 1, 1])
    
    # График жеста из датасета
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    dataset_data = dataset_tensor.numpy()
    
    if len(dataset_data) > 0:
        # Убираем NaN значения
        valid_mask = ~np.isnan(dataset_data).any(axis=2)
        valid_frames = []
        for frame in dataset_data:
            valid_frame = frame[~np.isnan(frame).any(axis=1)]
            if len(valid_frame) > 0:
                valid_frames.append(valid_frame)
        
        if valid_frames:
            all_points = np.vstack(valid_frames)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
            
            # Добавляем отступы
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            z_margin = (z_max - z_min) * 0.1
            
            ax2.set_xlim(x_min - x_margin, x_max + x_margin)
            ax2.set_ylim(y_min - y_margin, y_max + y_margin)
            ax2.set_zlim(z_min - z_margin, z_max + z_margin)
            
            # Рисуем точки
            ax2.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], 
                       c='red', s=20, alpha=0.7, label=f'Landmarks ({len(all_points)} точек)')
    
    ax2.set_xlabel('X координата')
    ax2.set_ylabel('Y координата')
    ax2.set_zlabel('Z координата')
    ax2.set_title(f"{dataset_title}\nКадров: {len(dataset_data)}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()

def analyze_similarity(captured_tensor: torch.Tensor, 
                      dataset_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Анализирует сходство между захваченным жестом и жестом из датасета
    
    Returns:
        similarity_metrics: Словарь с метриками сходства
    """
    print("\n📊 Анализ сходства жестов:")
    
    captured_data = captured_tensor.numpy()
    dataset_data = dataset_tensor.numpy()
    
    if len(captured_data) == 0 or len(dataset_data) == 0:
        print("❌ Недостаточно данных для анализа")
        return {}
    
    # Сравниваем размеры тензоров
    print(f"   Захваченный жест: {captured_tensor.shape}")
    print(f"   Жест из датасета: {dataset_tensor.shape}")
    
    # Проверяем соответствие форматов
    if captured_tensor.shape[1] != dataset_tensor.shape[1]:
        print(f"   ⚠️ Разное количество landmarks: {captured_tensor.shape[1]} vs {dataset_tensor.shape[1]}")
    else:
        print(f"   ✅ Количество landmarks совпадает: {captured_tensor.shape[1]}")
    
    if captured_tensor.shape[0] != dataset_tensor.shape[0]:
        print(f"   ⚠️ Разное количество кадров: {captured_tensor.shape[0]} vs {dataset_tensor.shape[0]}")
    else:
        print(f"   ✅ Количество кадров совпадает: {captured_tensor.shape[0]}")
    
    # Убираем NaN значения
    captured_valid = []
    for frame in captured_data:
        valid_frame = frame[~np.isnan(frame).any(axis=1)]
        if len(valid_frame) > 0:
            captured_valid.append(valid_frame)
    
    dataset_valid = []
    for frame in dataset_data:
        valid_frame = frame[~np.isnan(frame).any(axis=1)]
        if len(valid_frame) > 0:
            dataset_valid.append(valid_frame)
    
    if not captured_valid or not dataset_valid:
        print("❌ Нет валидных данных для анализа")
        return {}
    
    # Сравниваем диапазоны координат
    captured_all = np.vstack(captured_valid)
    dataset_all = np.vstack(dataset_valid)
    
    captured_ranges = {
        'x': (captured_all[:, 0].min(), captured_all[:, 0].max()),
        'y': (captured_all[:, 1].min(), captured_all[:, 1].max()),
        'z': (captured_all[:, 2].min(), captured_all[:, 2].max())
    }
    
    dataset_ranges = {
        'x': (dataset_all[:, 0].min(), dataset_all[:, 0].max()),
        'y': (dataset_all[:, 1].min(), dataset_all[:, 1].max()),
        'z': (dataset_all[:, 2].min(), dataset_all[:, 2].max())
    }
    
    print(f"   Диапазоны координат:")
    for coord in ['x', 'y', 'z']:
        print(f"     {coord.upper()}: захваченный {captured_ranges[coord][0]:.3f}-{captured_ranges[coord][1]:.3f}, "
              f"датасет {dataset_ranges[coord][0]:.3f}-{dataset_ranges[coord][1]:.3f}")
    
    # Вычисляем сходство диапазонов
    range_similarity = 0
    for coord in ['x', 'y', 'z']:
        captured_range = captured_ranges[coord][1] - captured_ranges[coord][0]
        dataset_range = dataset_ranges[coord][1] - dataset_ranges[coord][0]
        
        if captured_range > 0 and dataset_range > 0:
            min_range = min(captured_range, dataset_range)
            max_range = max(captured_range, dataset_range)
            range_similarity += min_range / max_range
    
    range_similarity /= 3
    
    print(f"   Сходство диапазонов: {range_similarity:.3f}")
    
    # Сравниваем количество кадров
    frame_similarity = min(len(captured_valid), len(dataset_valid)) / max(len(captured_valid), len(dataset_valid))
    print(f"   Сходство количества кадров: {frame_similarity:.3f}")
    
    # Сходство форматов (размеров тензоров)
    format_similarity = 0
    if captured_tensor.shape[1] == dataset_tensor.shape[1]:
        format_similarity += 0.5  # Совпадение landmarks
    if captured_tensor.shape[0] == dataset_tensor.shape[0]:
        format_similarity += 0.5  # Совпадение кадров
    
    print(f"   Сходство форматов: {format_similarity:.3f}")
    
    # Общая оценка сходства
    overall_similarity = (range_similarity + frame_similarity + format_similarity) / 3
    print(f"   Общее сходство: {overall_similarity:.3f}")
    
    if overall_similarity > 0.7:
        print("   ✅ Жесты похожи!")
    elif overall_similarity > 0.4:
        print("   ⚠️ Жесты частично похожи")
    else:
        print("   ❌ Жесты сильно отличаются")
    
    return {
        'range_similarity': range_similarity,
        'frame_similarity': frame_similarity,
        'format_similarity': format_similarity,
        'overall_similarity': overall_similarity
    }

# Примеры использования
if __name__ == "__main__":
    print("🎥 Захват жестов с веб-камеры")
    print("=" * 50)
    
    # Загружаем датасет для сравнения
    print("📁 Загружаем датасет для сравнения...")
    train_data, train_labels, test_data, test_labels, sign_mapping, classes = load_dataset(max_samples=5)
    
    if not train_data:
        print("❌ Не удалось загрузить датасет")
        exit()
    
    # Создаем захватчик с настройками для датасета
    print("🎯 Настройки захвата:")
    print("   - target_frames=16: ограничение кадров как в датасете")
    print("   - use_only_face=False: все landmarks (543 точки) - исправленный датасет")
    print("   - use_only_face=True: только face landmarks (468 точек) - старый формат")
    
    capture = MediaPipeCapture(
        camera_id=0,
        max_frames=50,
        fps=15,
        show_video=True,
        target_frames=16,  # Целевое количество кадров как в датасете
        use_only_face=False  # Используем все landmarks (исправленный датасет)
    )
    
    # Захватываем жест
    print("\n🎬 Начинаем захват жеста...")
    landmarks_data, timestamps = capture.start_capture()
    
    if not landmarks_data:
        print("❌ Не удалось захватить жест")
        exit()
    
    # Конвертируем в тензор
    captured_tensor = capture.convert_to_dataset_format()
    print(f"✅ Захвачен жест: {captured_tensor.shape}")
    
    # Сохраняем жест
    capture.save_gesture("my_gesture.json")
    
    # Выбираем случайный жест из датасета для сравнения
    import random
    random_idx = random.randint(0, len(train_data) - 1)
    dataset_tensor = train_data[random_idx]
    dataset_label = train_labels[random_idx]
    
    # Создаем обратный маппинг
    reverse_mapping = {v: k for k, v in sign_mapping.items()}
    dataset_sign_name = reverse_mapping[dataset_label]
    
    print(f"\n🔄 Сравниваем с жестом из датасета:")
    print(f"   Знак: '{dataset_sign_name}' (метка: {dataset_label})")
    print(f"   Размер: {dataset_tensor.shape}")
    
    # Анализируем сходство
    similarity_metrics = analyze_similarity(captured_tensor, dataset_tensor)
    
    # Визуализируем сравнение
    print("\n🖼️ Визуализация сравнения:")
    visualize_comparison(
        captured_tensor, 
        dataset_tensor,
        captured_title="Ваш захваченный жест",
        dataset_title=f"Жест из датасета: {dataset_sign_name}"
    )
    
    print("\n✅ Анализ завершен!")
    print("💡 Используйте функции:")
    print("   - MediaPipeCapture() - для захвата жестов")
    print("   - analyze_similarity() - для анализа сходства")
    print("   - visualize_comparison() - для визуализации сравнения")
    print("\n🎯 Для захвата нового жеста:")
    print("   # С полным набором landmarks (543 точки) - РЕКОМЕНДУЕТСЯ:")
    print("   capture = MediaPipeCapture(target_frames=16, use_only_face=False)")
    print("   landmarks, timestamps = capture.start_capture()")
    print("   tensor = capture.convert_to_dataset_format()")
    print("\n   # Только face landmarks (468 точек) - старый формат:")
    print("   capture = MediaPipeCapture(target_frames=16, use_only_face=True)")
    print("   landmarks, timestamps = capture.start_capture()")
    print("   tensor = capture.convert_to_dataset_format()")
    print("\n📊 Новые возможности:")
    print("   - target_frames: ограничение количества кадров")
    print("   - use_only_face: выбор типа landmarks")
    print("   - convert_to_dataset_format(): конвертация в формат датасета")
    print("   - normalize_landmarks(): нормализация данных")
    print("\n💡 Примечание:")
    print("   - Исправленный датасет теперь использует все landmarks (543 точки)")
    print("   - Структура: Face(468) + Pose(33) + Left Hand(21) + Right Hand(21)")
    print("   - Для распознавания жестов рук используйте use_only_face=False")
