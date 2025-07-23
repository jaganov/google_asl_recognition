import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import os
import sys
import platform
from datetime import datetime

# Импортируем модель и классы из step3
from step3_prepare_train import (
    ASLModel, 
    PreprocessingLayer, 
    AdaptiveDropout,
    TemporalConvBlock,
    BidirectionalLSTM,
    TemporalAttention,
    Conv1DBlock,
    TransformerBlock
)
from step2_prepare_dataset import load_dataset

class MacLiveASLRecognition:
    """Живое распознавание ASL жестов для macOS"""
    
    def __init__(self, 
                 model_path: str = None,
                 camera_id: int = 0,
                 target_frames: int = 16,
                 use_only_face: bool = False,
                 confidence_threshold: float = 0.8,
                 save_screenshots: bool = True):
        """
        Args:
            model_path: Путь к обученной модели (автоматически ищет если None)
            camera_id: ID камеры
            target_frames: Целевое количество кадров
            use_only_face: Использовать только face landmarks
            confidence_threshold: Порог уверенности для сохранения скриншота (0.8 = 80%)
            save_screenshots: Сохранять ли скриншоты высокоточных предсказаний
        """
        print("🍎 Инициализация ASL Recognition для macOS...")
        
        # Проверяем платформу
        self.is_mac = platform.system() == "Darwin"
        if self.is_mac:
            print("   Обнаружена macOS")
        
        self.camera_id = camera_id
        self.target_frames = target_frames
        self.use_only_face = use_only_face
        self.confidence_threshold = confidence_threshold
        self.save_screenshots = save_screenshots
        
        # Определяем базовую директорию (для standalone приложения)
        if getattr(sys, 'frozen', False):
            # Запущено как standalone приложение
            self.base_dir = Path(sys._MEIPASS)
        else:
            # Запущено как Python скрипт
            self.base_dir = Path(__file__).parent
        
        print(f"   Базовая директория: {self.base_dir}")
        
        # Автоматический поиск модели если не указана
        if model_path is None:
            model_path = self._find_latest_model()
        
        # Создаем папку для скриншотов в домашней директории пользователя
        self.screenshots_dir = Path.home() / "ASL_Screenshots"
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Счетчик сохраненных скриншотов для каждого жеста
        self.screenshot_counters = {}
        
        # Инициализация MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Принудительно используем CPU для Mac (без CUDA)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"   Используемое устройство: {self.device}")
        
        # Загружаем модель
        self.model, self.sign_mapping, self.classes = self._load_model(model_path)
        
        # Буфер для кадров
        self.frame_buffer = []
        self.max_buffer_size = 30
        
        print(f"🎯 Live ASL Recognition инициализирован:")
        print(f"   Устройство: {self.device}")
        print(f"   Модель: {model_path}")
        print(f"   Классов: {len(self.classes) if self.classes else 0}")
        print(f"   Целевых кадров: {target_frames}")
        print(f"   Только face landmarks: {use_only_face}")
        print(f"   Порог уверенности: {confidence_threshold:.1%}")
        print(f"   Сохранение скриншотов: {save_screenshots}")
        if save_screenshots:
            print(f"   Папка скриншотов: {self.screenshots_dir}")
    
    def _find_latest_model(self) -> str:
        """Автоматически находит последнюю модель"""
        models_dirs = [
            self.base_dir / "models",
            self.base_dir / "manual" / "models",
            Path("models"),
            Path("manual/models")
        ]
        
        latest_model = None
        latest_time = 0
        
        for models_dir in models_dirs:
            if models_dir.exists():
                print(f"   Поиск моделей в: {models_dir}")
                for model_file in models_dir.glob("*.pth"):
                    try:
                        file_time = model_file.stat().st_mtime
                        if file_time > latest_time:
                            latest_time = file_time
                            latest_model = str(model_file)
                            print(f"   Найдена модель: {model_file.name}")
                    except Exception as e:
                        print(f"   Ошибка при проверке {model_file}: {e}")
        
        if latest_model:
            print(f"   Выбрана последняя модель: {latest_model}")
            return latest_model
        else:
            print("   ⚠️ Модель не найдена, используем путь по умолчанию")
            return "models/asl_model_v20250720_080209.pth"
    
    def _load_model(self, model_path: str):
        """Загружает модель и маппинг знаков"""
        # Поиск маппинга знаков
        mapping_paths = [
            self.base_dir / "dataset25" / "sign_to_prediction_index_map.json",
            self.base_dir / "manual" / "dataset25" / "sign_to_prediction_index_map.json",
            Path("dataset25/sign_to_prediction_index_map.json"),
            Path("manual/dataset25/sign_to_prediction_index_map.json")
        ]
        
        sign_mapping = {}
        classes = []
        
        for mapping_path in mapping_paths:
            if mapping_path.exists():
                print(f"   Найден маппинг знаков: {mapping_path}")
                try:
                    with open(mapping_path, 'r') as f:
                        sign_mapping = json.load(f)
                    # Создаем список классов в правильном порядке
                    classes = [""] * len(sign_mapping)
                    for sign, idx in sign_mapping.items():
                        classes[idx] = sign
                    break
                except Exception as e:
                    print(f"   Ошибка при загрузке маппинга: {e}")
        
        if not sign_mapping:
            print(f"   ⚠️ Маппинг знаков не найден, используем значения по умолчанию")
            classes = [f"class_{i}" for i in range(25)]
        
        # Загружаем датасет для получения размерности входа (если доступен)
        input_dim = 744  # Значение по умолчанию
        try:
            train_data, train_labels, test_data, test_labels, _, _ = load_dataset(max_samples=1)
            sample_sequence = train_data[0]
            sample_preprocessed = PreprocessingLayer(max_len=384)(sample_sequence.unsqueeze(0))
            input_dim = sample_preprocessed.shape[-1]
            print(f"   Размерность входа после preprocessing: {input_dim}")
        except Exception as e:
            print(f"   ⚠️ Ошибка при загрузке датасета: {e}")
            print(f"   Используем размерность по умолчанию: {input_dim}")
        
        # Загружаем модель
        model_paths = [
            Path(model_path),
            self.base_dir / model_path,
            self.base_dir / "models" / Path(model_path).name,
            self.base_dir / "manual" / "models" / Path(model_path).name
        ]
        
        model = None
        for path in model_paths:
            if path.exists():
                print(f"   Загружаем модель из: {path}")
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Создаем совместимую модель
                    model = self._create_compatible_model_from_state_dict(input_dim, len(classes), state_dict)
                    model.load_state_dict(state_dict)
                    print(f"   ✅ Модель загружена успешно")
                    break
                except Exception as e:
                    print(f"   ❌ Ошибка при загрузке модели: {e}")
        
        if model is None:
            print(f"   ❌ Не удалось загрузить модель")
            return None, sign_mapping, classes
        
        model.eval()
        return model, sign_mapping, classes
    
    def _create_compatible_model_from_state_dict(self, input_dim: int, num_classes: int, state_dict: dict):
        """Создает совместимую модель на основе анализа state_dict"""
        class CompatibleASLModel(nn.Module):
            def __init__(self, input_dim, num_classes, max_len=384, dim=192):
                super().__init__()
                self.max_len = max_len
                self.dim = dim
                
                # Preprocessing
                self.preprocessing = PreprocessingLayer(max_len)
                
                # Stem with improved initialization
                self.stem_conv = nn.Linear(input_dim, dim, bias=False)
                self.stem_bn = nn.BatchNorm1d(dim, momentum=0.95)
                self.stem_dropout = AdaptiveDropout(initial_p=0.1, final_p=0.3, warmup_epochs=20)
                
                # Temporal Convolutional Network (TCN) - 3 blocks with different dilations
                self.tcn1 = TemporalConvBlock(dim, kernel_size=17, dilation=1, drop_rate=0.15)
                self.tcn2 = TemporalConvBlock(dim, kernel_size=17, dilation=2, drop_rate=0.2)
                self.tcn3 = TemporalConvBlock(dim, kernel_size=17, dilation=4, drop_rate=0.25)
                
                # Projection for attention pooling
                self.attention_projection = nn.Linear(dim, dim)
                
                # Bidirectional LSTM - 2 layers for better dependency capture
                self.lstm = BidirectionalLSTM(dim, hidden_dim=dim//2, num_layers=2, drop_rate=0.15)
                
                # Temporal Attention - more heads for better attention
                self.temporal_attention = TemporalAttention(dim, num_heads=8, drop_rate=0.15)
                
                # 1D CNN + Transformer blocks - 3 conv blocks
                ksize = 17
                self.conv1 = Conv1DBlock(dim, ksize, drop_rate=0.15)
                self.conv2 = Conv1DBlock(dim, ksize, drop_rate=0.2)
                self.conv3 = Conv1DBlock(dim, ksize, drop_rate=0.25)
                self.transformer1 = TransformerBlock(dim, expand=2, drop_rate=0.15)
                
                # Top layers with improved pooling
                self.top_conv = nn.Linear(dim, dim)
                self.top_bn = nn.BatchNorm1d(dim, momentum=0.95)
                self.adaptive_dropout = AdaptiveDropout(initial_p=0.2, final_p=0.5, warmup_epochs=25)
                
                # Improved pooling - add attention pooling
                self.temporal_pool1 = nn.AdaptiveAvgPool1d(1)
                self.temporal_pool2 = nn.AdaptiveMaxPool1d(1)
                self.attention_pool = nn.MultiheadAttention(dim, num_heads=4, batch_first=True, dropout=0.1)
                
                # Final classifier with improved architecture
                self.classifier = nn.Sequential(
                    nn.Linear(dim * 3, dim),  # dim*3 for avg + max + attention pooling
                    nn.BatchNorm1d(dim),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(dim, num_classes)
                )
                
                # Инициализация весов
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            init.zeros_(m.bias)
                    elif isinstance(m, nn.Conv1d):
                        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm1d):
                        init.ones_(m.weight)
                        init.zeros_(m.bias)
                    elif isinstance(m, nn.LSTM):
                        for name, param in m.named_parameters():
                            if 'weight' in name:
                                init.orthogonal_(param)
                            elif 'bias' in name:
                                init.zeros_(param)
            
            def forward(self, x):
                # x: (batch, seq, landmarks, 3)
                
                # Preprocessing
                x = self.preprocessing(x)  # (batch, seq, features)
                
                # Stem with adaptive dropout
                x = self.stem_conv(x)
                x = self.stem_bn(x.transpose(1, 2)).transpose(1, 2)
                x = self.stem_dropout(x)
                
                # TCN blocks - 3 blocks with different dilations
                x = self.tcn1(x)
                x = self.tcn2(x)
                x = self.tcn3(x)
                
                # Bidirectional LSTM - 2 layers
                x = self.lstm(x)
                
                # Temporal Attention - more heads
                x = self.temporal_attention(x)
                
                # 1D CNN + Transformer - 3 conv blocks
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.transformer1(x)
                
                # Top layers with improved pooling
                x = self.top_conv(x)  # (batch, seq, dim)
                x = self.top_bn(x.transpose(1, 2)).transpose(1, 2)
                x = self.adaptive_dropout(x)
                
                # Improved temporal pooling
                x_transposed = x.transpose(1, 2)  # (batch, dim, seq)
                
                # Global average pooling
                global_avg = self.temporal_pool1(x_transposed).squeeze(-1)  # (batch, dim)
                
                # Global max pooling
                global_max = self.temporal_pool2(x_transposed).squeeze(-1)  # (batch, dim)
                
                # Attention pooling - use projection for stability
                x_for_attn = self.attention_projection(x)  # x already has dimension (batch, seq, dim)
                
                attn_out, _ = self.attention_pool(x_for_attn, x_for_attn, x_for_attn)
                global_attn = torch.mean(attn_out, dim=1)  # (batch, dim)
                
                # Combine pooling results
                x_pooled = torch.cat([global_avg, global_max, global_attn], dim=1)  # (batch, dim*3)
                
                x = self.classifier(x_pooled)
                
                return x
        
        return CompatibleASLModel(input_dim, num_classes, max_len=384, dim=192).to(self.device)
    
    def _extract_landmarks(self, results) -> Optional[np.ndarray]:
        """Извлекает landmarks из результатов MediaPipe"""
        landmarks = []
        
        if self.use_only_face:
            if results.face_landmarks:
                for landmark in results.face_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
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
        """Рисует landmarks на кадре"""
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
    
    def _normalize_landmarks(self, landmarks_data: List[np.ndarray]) -> List[np.ndarray]:
        """Нормализует landmarks"""
        if not landmarks_data:
            return []
        
        # Ограничиваем количество кадров
        if len(landmarks_data) > self.target_frames:
            indices = np.linspace(0, len(landmarks_data) - 1, self.target_frames, dtype=int)
            landmarks_data = [landmarks_data[i] for i in indices]
        
        # Проверяем количество landmarks
        # Оригинальная модель ожидает 543 landmarks (468 face + 33 pose + 21 left hand + 21 right hand)
        expected_landmarks = 543
        
        # Приводим к правильному количеству landmarks
        normalized_data = []
        for frame in landmarks_data:
            if len(frame) >= expected_landmarks:
                normalized_frame = frame[:expected_landmarks]
            else:
                # Дополняем нулями если не хватает landmarks
                normalized_frame = np.zeros((expected_landmarks, 3))
                normalized_frame[:len(frame)] = frame
            
            normalized_data.append(normalized_frame)
        
        return normalized_data
    
    def _predict_gesture(self, landmarks_data: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Предсказывает жест"""
        if not landmarks_data or len(landmarks_data) < 5 or self.model is None:
            return []
        
        # Нормализуем данные
        normalized_data = self._normalize_landmarks(landmarks_data)
        
        if not normalized_data:
            return []
        
        try:
            # Конвертируем в тензор в правильном формате
            # Формат: (batch, seq, landmarks, 3)
            tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)
            
            # Предсказываем
            with torch.no_grad():
                predictions = self.model(tensor.to(self.device))
                probabilities = torch.softmax(predictions, dim=1)
            
            # Получаем топ-3 предсказания
            top_probs, top_indices = torch.topk(probabilities, min(3, len(self.classes)), dim=1)
            
            results = []
            for i in range(min(3, len(self.classes))):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                
                # Получаем название знака
                if idx < len(self.classes) and self.classes[idx]:
                    sign_name = self.classes[idx]
                else:
                    sign_name = f"Unknown_{idx}"
                
                results.append((sign_name, prob))
            
            return results
        except Exception as e:
            print(f"   ⚠️ Ошибка при предсказании: {e}")
            return []
    
    def _save_screenshot(self, frame: np.ndarray, predictions: List[Tuple[str, float]]):
        """Сохраняет скриншот с предсказаниями"""
        if not self.save_screenshots or not predictions:
            return
        
        # Проверяем, есть ли предсказание с высокой уверенностью
        top_prediction = predictions[0]
        sign_name, confidence = top_prediction
        
        if confidence < self.confidence_threshold:
            return
        
        try:
            # Создаем копию кадра для скриншота
            screenshot = frame.copy()
            
            # Добавляем информацию о предсказаниях на скриншот
            y_offset = 30
            cv2.putText(screenshot, "ASL Recognition Results:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 40
            
            # Добавляем временную метку
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(screenshot, f"Time: {timestamp}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
            
            # Добавляем топ-3 предсказания
            cv2.putText(screenshot, "Top 3 Predictions:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            
            for i, (sign, prob) in enumerate(predictions):
                # Цвет в зависимости от уверенности
                if prob >= self.confidence_threshold:
                    color = (0, 255, 0)  # Зеленый для высокоточных
                elif i == 0:
                    color = (255, 255, 0)  # Желтый для первого
                else:
                    color = (0, 255, 255)  # Голубой для остальных
                
                text = f"{i+1}. {sign}: {prob:.3f} ({prob:.1%})"
                cv2.putText(screenshot, text, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
            
            # Добавляем информацию о настройках
            y_offset += 10
            cv2.putText(screenshot, f"Confidence Threshold: {self.confidence_threshold:.1%}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
            cv2.putText(screenshot, f"Frames: {len(self.frame_buffer)}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Генерируем имя файла
            if sign_name not in self.screenshot_counters:
                self.screenshot_counters[sign_name] = 0
            self.screenshot_counters[sign_name] += 1
            
            # Очищаем имя файла от недопустимых символов
            safe_sign_name = "".join(c for c in sign_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_sign_name = safe_sign_name.replace(' ', '_')
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_sign_name}_{timestamp_str}_{self.screenshot_counters[sign_name]:03d}.jpg"
            filepath = self.screenshots_dir / filename
            
            # Сохраняем скриншот
            success = cv2.imwrite(str(filepath), screenshot)
            if success:
                print(f"📸 Сохранен скриншот: {filename} (уверенность: {confidence:.1%})")
            else:
                print(f"❌ Ошибка сохранения скриншота: {filename}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении скриншота: {e}")
    
    def start_recognition(self):
        """Запускает живое распознавание"""
        print("\n🎬 Начинаем живое распознавание ASL жестов на macOS...")
        print("   Нажмите 'q' для выхода")
        
        if self.model is None:
            print("❌ Модель не загружена. Невозможно начать распознавание.")
            return
        
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("❌ Ошибка: Не удалось открыть камеру")
            print("   Попробуйте изменить camera_id (например, 1 вместо 0)")
            return
        
        # Настройка камеры
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.frame_buffer = []
        last_prediction_time = time.time()
        prediction_interval = 0.5  # Предсказание каждые 0.5 секунд
        
        try:
            while True:
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
                    self.frame_buffer.append(frame_landmarks)
                    
                    # Ограничиваем размер буфера
                    if len(self.frame_buffer) > self.max_buffer_size:
                        self.frame_buffer.pop(0)
                
                # Рисуем landmarks на кадре
                annotated_frame = self._draw_landmarks(frame, results)
                
                # Предсказываем жест каждые 0.5 секунд
                current_time = time.time()
                if current_time - last_prediction_time > prediction_interval and len(self.frame_buffer) >= 5:
                    predictions = self._predict_gesture(self.frame_buffer)
                    last_prediction_time = current_time
                    
                    # Сохраняем скриншот если есть высокоточное предсказание
                    if predictions and self.save_screenshots:
                        self._save_screenshot(annotated_frame, predictions)
                    
                    # Отображаем предсказания на кадре
                    if predictions:
                        y_offset = 30
                        cv2.putText(annotated_frame, "Top 3 Predictions:", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                        
                        for i, (sign, prob) in enumerate(predictions):
                            # Цвет в зависимости от уверенности
                            if prob >= self.confidence_threshold:
                                color = (0, 255, 0)  # Зеленый для высокоточных
                                thickness = 3  # Толще для высокоточных
                            elif i == 0:
                                color = (255, 255, 0)  # Желтый для первого
                                thickness = 2
                            else:
                                color = (0, 255, 255)  # Голубой для остальных
                                thickness = 2
                            
                            text = f"{i+1}. {sign}: {prob:.3f} ({prob:.1%})"
                            cv2.putText(annotated_frame, text, 
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
                            y_offset += 25
                        
                        # Показываем статус сохранения скриншотов
                        if self.save_screenshots:
                            top_confidence = predictions[0][1]
                            if top_confidence >= self.confidence_threshold:
                                cv2.putText(annotated_frame, "📸 Screenshot saved!", 
                                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.putText(annotated_frame, f"Waiting for {self.confidence_threshold:.1%} confidence...", 
                                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Показываем информацию на кадре
                y_bottom = annotated_frame.shape[0] - 100
                cv2.putText(annotated_frame, f"Device: {self.device}", 
                           (10, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_bottom += 20
                cv2.putText(annotated_frame, f"Frames: {len(self.frame_buffer)}", 
                           (10, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_bottom += 20
                cv2.putText(annotated_frame, f"Confidence Threshold: {self.confidence_threshold:.1%}", 
                           (10, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_bottom += 20
                cv2.putText(annotated_frame, "Press 'q' to quit", 
                           (10, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('ASL Recognition for macOS', annotated_frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("   Выход...")
                    break
        
        except KeyboardInterrupt:
            print("   Прерывание пользователем...")
        except Exception as e:
            print(f"❌ Ошибка во время работы: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print("✅ Распознавание завершено")

# Пример использования
if __name__ == "__main__":
    print("🎥 Живое распознавание ASL жестов для macOS")
    print("=" * 50)
    
    # Создаем распознаватель
    recognizer = MacLiveASLRecognition(
        model_path=None,  # Автоматический поиск последней модели
        camera_id=0,
        target_frames=16,
        use_only_face=False,  # Используем все landmarks для лучшего распознавания
        confidence_threshold=0.8,  # Сохранять скриншоты при уверенности >= 80%
        save_screenshots=True  # Включить сохранение скриншотов
    )
    
    # Запускаем распознавание
    recognizer.start_recognition() 