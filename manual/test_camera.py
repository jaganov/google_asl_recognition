import cv2
import mediapipe as mp
import numpy as np

def test_camera():
    """
    Простой тест камеры и MediaPipe
    """
    print("🎥 Тестирование камеры и MediaPipe...")
    
    # Инициализация MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Открываем камеру
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Ошибка: Не удалось открыть камеру")
        print("   Попробуйте изменить camera_id на 1 или 2")
        return False
    
    print("✅ Камера открыта успешно")
    print("   Нажмите 'q' для выхода")
    
    frame_count = 0
    
    try:
        while True:  # Работаем непрерывно до нажатия 'q'
            ret, frame = cap.read()
            if not ret:
                print("❌ Ошибка чтения кадра")
                break
            
            # Конвертируем BGR в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Обрабатываем кадр через MediaPipe
            results = holistic.process(rgb_frame)
            
            # Рисуем landmarks
            annotated_frame = frame.copy()
            
            # Face landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            
            # Pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Left hand landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Right hand landmarks
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Показываем информацию
            cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Показываем статус обнаружения
            face_detected = "Face: ✓" if results.face_landmarks else "Face: ✗"
            pose_detected = "Pose: ✓" if results.pose_landmarks else "Pose: ✗"
            left_hand_detected = "Left: ✓" if results.left_hand_landmarks else "Left: ✗"
            right_hand_detected = "Right: ✓" if results.right_hand_landmarks else "Right: ✗"
            
            cv2.putText(annotated_frame, face_detected, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, pose_detected, 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, left_hand_detected, 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, right_hand_detected, 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Camera Test', annotated_frame)
            
            frame_count += 1
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("   Выход по запросу пользователя")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"✅ Камера остановлена: {frame_count} кадров обработано")
    print("   Если вы видели landmarks на видео, камера работает корректно")
    
    return True

if __name__ == "__main__":
    test_camera() 