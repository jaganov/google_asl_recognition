# 🎥 Live Recognition Guide

This guide covers real-time ASL gesture recognition using the trained model and MediaPipe for landmark extraction.

## 🎯 Overview

The live recognition system (`step5_live_recognition.py`) provides real-time ASL gesture recognition through webcam input:

- **Real-time Processing**: <50ms per prediction
- **MediaPipe Integration**: 543 landmarks per frame
- **Visual Feedback**: Live landmark visualization
- **Screenshot Capture**: Save high-confidence predictions
- **Gesture Recognition**: 25 ASL gestures

## 🏗️ System Architecture

### Components
1. **MediaPipe Holistic**: Face, pose, and hand landmark extraction
2. **Landmark Processing**: Normalization and feature extraction
3. **Model Inference**: Real-time prediction using trained model
4. **Visualization**: Live landmark drawing and prediction display
5. **Screenshot System**: Automatic capture of high-confidence predictions

### Data Flow
```
Webcam Input
    ↓
MediaPipe Holistic
    ↓
Landmark Extraction (543 points)
    ↓
Landmark Selection (62 key points)
    ↓
Normalization & Preprocessing
    ↓
Model Inference
    ↓
Prediction Display
    ↓
Screenshot (if high confidence)
```

## 🔧 MediaPipe Integration

### Holistic Model Configuration
```python
self.holistic = self.mp_holistic.Holistic(
    min_detection_confidence=0.5,    # Minimum detection confidence
    min_tracking_confidence=0.5,     # Minimum tracking confidence
    model_complexity=1               # Model complexity (0, 1, 2)
)
```

### Landmark Structure
- **Face Landmarks**: 468 points (0-467)
- **Pose Landmarks**: 33 points (468-500)
- **Left Hand**: 21 points (501-521)
- **Right Hand**: 21 points (522-542)
- **Total**: 543 landmarks per frame

### Key Landmark Selection
```python
# Face landmarks (eyes, nose, lips)
face_landmarks = [33, 133, 362, 263, 61, 291, 199, 419, 17, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]

# Hand landmarks (key hand points)
left_hand = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521]
right_hand = [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

point_landmarks = face_landmarks + left_hand + right_hand  # 62 total landmarks
```

## 🚀 Usage

### Basic Usage
```bash
cd manual
python step5_live_recognition.py
```

### Advanced Configuration
```python
# Initialize with custom parameters
recognition = LiveASLRecognition(
    model_path="models/asl_model_v20250720_080209.pth",
    camera_id=0,                    # Camera device ID
    target_frames=16,               # Target sequence length
    use_only_face=False,            # Use only face landmarks
    confidence_threshold=0.8,       # Screenshot threshold (80%)
    save_screenshots=True           # Enable screenshot capture
)

# Start recognition
recognition.start_recognition()
```

### Command Line Options
```bash
# Test camera first
python test_camera.py

# Run with different camera
python step5_live_recognition.py --camera 1

# Run with face-only mode
python step5_live_recognition.py --face-only

# Run with custom confidence threshold
python step5_live_recognition.py --confidence 0.9
```

## 🎮 Controls

### Keyboard Controls
- **'q'**: Stop recognition and exit
- **'s'**: Save current screenshot manually
- **'r'**: Reset frame buffer
- **'h'**: Show/hide help information
- **'c'**: Toggle confidence display
- **'l'**: Toggle landmark visualization

### Visual Feedback
- **Landmark Drawing**: Real-time landmark visualization
- **Prediction Display**: Current gesture and confidence
- **Frame Buffer**: Number of frames collected
- **FPS Counter**: Real-time performance metrics

## 📊 Recognition Process

### 1. Frame Capture
```python
def capture_frame(self):
    """
    Capture frame from webcam
    """
    ret, frame = self.cap.read()
    if not ret:
        return None
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    return frame
```

### 2. Landmark Extraction
```python
def _extract_landmarks(self, results) -> Optional[np.ndarray]:
    """
    Extract landmarks from MediaPipe results
    """
    landmarks_data = []
    
    # Extract face landmarks
    if results.face_landmarks:
        face_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        landmarks_data.append(face_landmarks)
    
    # Extract pose landmarks
    if results.pose_landmarks:
        pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        landmarks_data.append(pose_landmarks)
    
    # Extract hand landmarks
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        landmarks_data.append(left_hand)
    
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        landmarks_data.append(right_hand)
    
    # Combine all landmarks
    if landmarks_data:
        return np.concatenate(landmarks_data, axis=0)
    
    return None
```

### 3. Landmark Normalization
```python
def _normalize_landmarks(self, landmarks_data: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize landmarks relative to nose
    """
    normalized_data = []
    
    for landmarks in landmarks_data:
        if landmarks is None or len(landmarks) == 0:
            continue
        
        # Select key landmarks (62 points)
        key_landmarks = landmarks[self.point_landmarks]
        
        # Normalize relative to nose (index 1 in face landmarks)
        nose_coords = key_landmarks[1:2]  # Nose coordinates
        normalized = key_landmarks - nose_coords
        
        normalized_data.append(normalized)
    
    return normalized_data
```

### 4. Model Prediction
```python
def _predict_gesture(self, landmarks_data: List[np.ndarray]) -> List[Tuple[str, float]]:
    """
    Predict ASL gesture from landmarks
    """
    if len(landmarks_data) < self.target_frames:
        return []
    
    # Prepare input tensor
    input_tensor = torch.tensor(landmarks_data, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    input_tensor = input_tensor.to(self.device)
    
    # Model inference
    with torch.no_grad():
        outputs = self.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)
    
    predictions = []
    for i in range(top_indices.shape[1]):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        gesture = self.classes[idx]
        predictions.append((gesture, prob))
    
    return predictions
```

## 📸 Screenshot System

### Automatic Capture
```python
def _save_screenshot(self, frame: np.ndarray, predictions: List[Tuple[str, float]]):
    """
    Save screenshot for high-confidence predictions
    """
    if not predictions:
        return
    
    top_gesture, top_confidence = predictions[0]
    
    # Check confidence threshold
    if top_confidence < self.confidence_threshold:
        return
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    # Update counter for this gesture
    if top_gesture not in self.screenshot_counters:
        self.screenshot_counters[top_gesture] = 0
    self.screenshot_counters[top_gesture] += 1
    
    filename = f"{top_gesture}_{timestamp}_{self.screenshot_counters[top_gesture]:03d}.jpg"
    filepath = self.screenshots_dir / filename
    
    # Add prediction text to image
    annotated_frame = frame.copy()
    text = f"{top_gesture}: {top_confidence:.1%}"
    cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save image
    cv2.imwrite(str(filepath), annotated_frame)
    print(f"📸 Screenshot saved: {filename}")
```

### Screenshot Organization
```
screenshots/
├── hello_20250721_134139_001.jpg
├── hello_20250721_134139_002.jpg
├── please_20250721_134145_001.jpg
├── thankyou_20250721_134150_001.jpg
└── ...
```

## 🎯 Recognized Gestures

The system recognizes 25 ASL gestures in real-time:

| Gesture | Category | Example Usage |
|---------|----------|---------------|
| hello | Greetings | Wave hello |
| please | Courtesy | Polite request |
| thankyou | Courtesy | Express gratitude |
| bye | Greetings | Wave goodbye |
| mom | Family | Point to mother |
| dad | Family | Point to father |
| boy | Family | Indicate male |
| girl | Family | Indicate female |
| man | Family | Adult male |
| child | Family | Young person |
| drink | Actions | Drinking motion |
| sleep | Actions | Sleeping gesture |
| go | Actions | Movement gesture |
| happy | Emotions | Smile gesture |
| sad | Emotions | Downward motion |
| hungry | Emotions | Stomach gesture |
| thirsty | Emotions | Throat gesture |
| sick | Emotions | Illness gesture |
| bad | Emotions | Negative gesture |
| red | Colors | Red color sign |
| blue | Colors | Blue color sign |
| green | Colors | Green color sign |
| yellow | Colors | Yellow color sign |
| black | Colors | Black color sign |
| white | Colors | White color sign |

## 🔧 Performance Optimization

### RTX4070 Optimizations
```python
# Enable TensorFloat-32 for faster computation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN benchmark for optimized algorithms
torch.backends.cudnn.benchmark = True

# Use mixed precision for inference
with torch.cuda.amp.autocast():
    outputs = self.model(input_tensor)
```

### Memory Management
```python
# Clear frame buffer periodically
if len(self.frame_buffer) > self.max_buffer_size:
    self.frame_buffer = self.frame_buffer[-self.max_buffer_size:]

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## 📊 Performance Metrics

### Expected Performance
- **Inference Time**: <50ms per prediction
- **FPS**: 20-30 FPS on RTX4070
- **Memory Usage**: ~2-4GB GPU memory
- **Accuracy**: ~76% on test set
- **Latency**: <100ms end-to-end

### Monitoring
```python
# FPS calculation
fps = 1.0 / (time.time() - self.last_frame_time)
self.last_frame_time = time.time()

# Display performance metrics
cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Camera Not Found
```bash
# Test camera availability
python test_camera.py

# Try different camera ID
python step5_live_recognition.py --camera 1
```

#### 2. Low FPS
```python
# Reduce target frames
target_frames = 8  # Instead of 16

# Use face-only mode
use_only_face = True

# Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

#### 3. Poor Recognition
```python
# Increase confidence threshold
confidence_threshold = 0.9

# Ensure good lighting
# Position camera properly
# Use clear hand gestures
```

#### 4. Model Loading Issues
```python
# Check model file exists
model_path = "models/asl_model_v20250720_080209.pth"
if not Path(model_path).exists():
    print(f"Model not found: {model_path}")
    exit(1)

# Verify model compatibility
try:
    model = torch.load(model_path, map_location=device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
```

## 📚 Related Documentation

- **[Training Guide](training.md)** - How to train the model
- **[Model Architecture](architecture.md)** - Model design details
- **[Data Preparation](data-preparation.md)** - Dataset preparation
- **[Installation Guide](installation.md)** - Setup instructions
- **[RTX4070 Optimizations](rtx4070-optimizations.md)** - Performance optimizations

---

**Ready for real-time ASL recognition! 🎥** 