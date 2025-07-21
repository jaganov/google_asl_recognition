# 📊 Data Preparation Guide

This guide covers the complete data preparation process for the ASL recognition project, from the original Google ASL Signs dataset to the final PyTorch tensors ready for training.

## 🎯 Overview

The data preparation pipeline transforms the Google ASL Signs dataset into a format optimized for deep learning training:

1. **Dataset Extraction**: Extract 25 most common ASL gestures
2. **Data Splitting**: Create train/test splits by participant
3. **Preprocessing**: Convert landmarks to PyTorch tensors
4. **Feature Engineering**: Add motion features and normalization

## 📁 Dataset Structure

### Original Dataset
```
data/google_asl_signs/
├── train.csv                    # Main training data
├── sign_to_prediction_index_map.json  # Sign mapping
└── train_landmark_files/        # Parquet files with landmarks
    ├── 10001.parquet           # Participant 1, sequence 1
    ├── 10002.parquet           # Participant 1, sequence 2
    └── ...
```

### Processed Dataset
```
manual/
├── dataset25/                   # Extracted 25 signs
│   ├── train.csv
│   ├── sign_to_prediction_index_map.json
│   └── train_landmark_files/
├── dataset25_split/             # Train/test split
│   ├── train/
│   │   ├── train.csv
│   │   ├── sign_to_prediction_index_map.json
│   │   └── train_landmark_files/
│   └── test/
│       ├── train.csv
│       ├── sign_to_prediction_index_map.json
│       └── train_landmark_files/
```

## 🎯 Recognized Gestures (25 ASL Signs)

The project focuses on 25 most common ASL gestures:

| Index | Gesture | Category |
|-------|---------|----------|
| 0 | hello | Greetings |
| 1 | please | Courtesy |
| 2 | thankyou | Courtesy |
| 3 | bye | Greetings |
| 4 | mom | Family |
| 5 | dad | Family |
| 6 | boy | Family |
| 7 | girl | Family |
| 8 | man | Family |
| 9 | child | Family |
| 10 | drink | Actions |
| 11 | sleep | Actions |
| 12 | go | Actions |
| 13 | happy | Emotions |
| 14 | sad | Emotions |
| 15 | hungry | Emotions |
| 16 | thirsty | Emotions |
| 17 | sick | Emotions |
| 18 | bad | Emotions |
| 19 | red | Colors |
| 20 | blue | Colors |
| 21 | green | Colors |
| 22 | yellow | Colors |
| 23 | black | Colors |
| 24 | white | Colors |

## 🔧 Data Preparation Scripts

### 1. Step 1: Extract Words (`step1_extract_words.py`)

Extracts the 25 target gestures from the full dataset.

```python
# Extract 25 signs from the full dataset
target_signs = [
    'hello', 'please', 'thankyou', 'bye', 'mom', 'dad', 'boy', 'girl', 'man', 'child',
    'drink', 'sleep', 'go', 'happy', 'sad', 'hungry', 'thirsty', 'sick', 'bad',
    'red', 'blue', 'green', 'yellow', 'black', 'white'
]

# Filter dataset to include only target signs
filtered_df = df[df['sign'].isin(target_signs)]
```

**Output**: `manual/dataset25/` directory with filtered data

### 2. Step 1.2: Split Train/Test (`step1.2_split_train_test.py`)

Creates train/test splits by participant to avoid data leakage.

```python
# Group by participant
participant_groups = df.groupby('participant_id')

# Split participants 80/20
train_participants = list(participant_groups.groups.keys())[:int(0.8 * len(participant_groups))]
test_participants = list(participant_groups.groups.keys())[int(0.8 * len(participant_groups)):]

# Create train/test splits
train_df = df[df['participant_id'].isin(train_participants)]
test_df = df[df['participant_id'].isin(test_participants)]
```

**Output**: `manual/dataset25_split/train/` and `manual/dataset25_split/test/`

### 3. Step 2: Prepare Dataset (`step2_prepare_dataset.py`)

Converts parquet landmark files into PyTorch tensors with preprocessing.

## 🏗️ Preprocessing Pipeline

### 1. Landmark Loading

```python
def _parquet_to_tensor_optimized(landmarks_df: pd.DataFrame, max_len: int) -> torch.Tensor:
    """
    Convert parquet landmarks to PyTorch tensor with optimization
    """
    # Extract landmark coordinates
    landmarks = landmarks_df[['x', 'y', 'z']].values
    
    # Reshape to (frames, landmarks, 3)
    num_frames = len(landmarks_df) // 543  # 543 landmarks per frame
    landmarks = landmarks.reshape(num_frames, 543, 3)
    
    # Pad or truncate to max_len
    if num_frames > max_len:
        landmarks = landmarks[:max_len]
    elif num_frames < max_len:
        padding = np.zeros((max_len - num_frames, 543, 3))
        landmarks = np.vstack([landmarks, padding])
    
    return torch.tensor(landmarks, dtype=torch.float32)
```

### 2. Landmark Selection

Selects 62 key landmarks for efficient processing:

```python
# Face landmarks (eyes, nose, lips)
face_landmarks = [33, 133, 362, 263, 61, 291, 199, 419, 17, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]

# Hand landmarks (key hand points)
left_hand = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521]
right_hand = [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

point_landmarks = face_landmarks + left_hand + right_hand  # 62 total landmarks
```

### 3. Normalization

Normalizes landmarks relative to the nose landmark:

```python
def normalize_landmarks(landmarks: torch.Tensor) -> torch.Tensor:
    """
    Normalize landmarks relative to nose landmark (index 1)
    """
    # Get nose coordinates
    nose_coords = landmarks[:, 1:2, :]  # Shape: (frames, 1, 3)
    
    # Normalize all landmarks relative to nose
    normalized = landmarks - nose_coords
    
    return normalized
```

### 4. Motion Features

Computes motion features for better temporal modeling:

```python
def compute_motion_features(x: torch.Tensor) -> torch.Tensor:
    """
    Compute motion features: velocity, acceleration, relative motion
    """
    # x: (batch, seq, landmarks, 3)
    batch_size, seq_len, num_landmarks, _ = x.shape
    
    # Velocity (lag1)
    dx = torch.zeros_like(x)
    dx[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
    
    # Acceleration (lag2)
    dx2 = torch.zeros_like(x)
    dx2[:, 2:, :, :] = dx[:, 2:, :, :] - dx[:, 1:-1, :, :]
    
    # Relative motion (lag3)
    dx3 = torch.zeros_like(x)
    dx3[:, 3:, :, :] = x[:, 3:, :, :] - x[:, :-3, :, :]
    
    # Concatenate features
    features = torch.cat([x, dx, dx2, dx3], dim=-1)  # Shape: (batch, seq, landmarks, 12)
    
    return features
```

## 📊 Dataset Statistics

### Data Distribution

```python
def get_dataset_statistics(train_data, test_data, sign_mapping):
    """
    Analyze dataset statistics
    """
    print("📊 Dataset Statistics:")
    print(f"   Training sequences: {len(train_data)}")
    print(f"   Test sequences: {len(test_data)}")
    print(f"   Total sequences: {len(train_data) + len(test_data)}")
    print(f"   Number of classes: {len(sign_mapping)}")
    
    # Sequence length distribution
    train_lengths = [seq.shape[0] for seq in train_data]
    test_lengths = [seq.shape[0] for seq in test_data]
    
    print(f"   Average sequence length (train): {np.mean(train_lengths):.1f}")
    print(f"   Average sequence length (test): {np.mean(test_lengths):.1f}")
    print(f"   Max sequence length: {max(max(train_lengths), max(test_lengths))}")
```

### Expected Statistics
- **Training sequences**: ~8,000-12,000
- **Test sequences**: ~2,000-3,000
- **Average sequence length**: ~150-200 frames
- **Maximum sequence length**: 384 frames
- **Landmarks per frame**: 62 (selected from 543)
- **Features per landmark**: 12 (3 coords + 9 motion features)

## 🔧 Usage

### 1. Extract 25 Signs
```bash
cd manual
python step1_extract_words.py
```

### 2. Create Train/Test Split
```bash
python step1.2_split_train_test.py
```

### 3. Prepare PyTorch Dataset
```bash
python step2_prepare_dataset.py
```

### 4. Verify Dataset
```python
from step2_prepare_dataset import load_dataset, get_dataset_statistics

# Load dataset
train_data, train_labels, test_data, test_labels, sign_mapping, classes = load_dataset(
    data_dir="dataset25_split",
    max_len=384,
    max_samples=None
)

# Get statistics
get_dataset_statistics(train_data, test_data, sign_mapping)
```

## 📈 Data Augmentation

The training script includes comprehensive data augmentation:

### 1. Temporal Augmentation
```python
def temporal_resample(self, x, target_length=None):
    """
    Resample sequence to different length
    """
    if target_length is None:
        target_length = random.randint(int(len(x) * 0.8), int(len(x) * 1.2))
    
    # Linear interpolation
    indices = torch.linspace(0, len(x) - 1, target_length)
    resampled = torch.stack([x[int(i)] for i in indices])
    
    return resampled
```

### 2. Spatial Augmentation
```python
def random_affine(self, x, max_scale=0.02, max_shift=0.01, max_rotate=2):
    """
    Apply random affine transformations
    """
    # Random scaling
    scale = 1 + random.uniform(-max_scale, max_scale)
    
    # Random translation
    shift_x = random.uniform(-max_shift, max_shift)
    shift_y = random.uniform(-max_shift, max_shift)
    
    # Random rotation
    angle = random.uniform(-max_rotate, max_rotate)
    
    # Apply transformations
    # ... implementation details
```

### 3. Temporal Masking
```python
def random_masking(self, x, mask_ratio=0.05):
    """
    Randomly mask temporal frames
    """
    seq_len = x.shape[0]
    mask_indices = random.sample(range(seq_len), int(seq_len * mask_ratio))
    
    for idx in mask_indices:
        x[idx] = 0  # Zero out masked frames
    
    return x
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Memory Issues
```python
# Reduce max_samples for testing
train_data, train_labels, test_data, test_labels, sign_mapping, classes = load_dataset(
    data_dir="dataset25_split",
    max_len=384,
    max_samples=1000  # Limit for testing
)
```

#### 2. Missing Files
```bash
# Check if dataset files exist
ls manual/dataset25_split/train/
ls manual/dataset25_split/test/
```

#### 3. Inconsistent Data
```python
# Verify data integrity
for i, (data, label) in enumerate(zip(train_data, train_labels)):
    if torch.isnan(data).any():
        print(f"NaN found in training sample {i}")
    if torch.isinf(data).any():
        print(f"Inf found in training sample {i}")
```

## 📊 Data Quality Checks

### 1. Landmark Validation
```python
def validate_landmarks(landmarks):
    """
    Validate landmark data quality
    """
    # Check for NaN values
    if torch.isnan(landmarks).any():
        return False
    
    # Check for infinite values
    if torch.isinf(landmarks).any():
        return False
    
    # Check for reasonable coordinate ranges
    if torch.abs(landmarks).max() > 1000:
        return False
    
    return True
```

### 2. Sequence Length Analysis
```python
def analyze_sequence_lengths(data):
    """
    Analyze sequence length distribution
    """
    lengths = [seq.shape[0] for seq in data]
    
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Std length: {np.std(lengths):.1f}")
    
    # Plot distribution
    plt.hist(lengths, bins=50)
    plt.title("Sequence Length Distribution")
    plt.xlabel("Length (frames)")
    plt.ylabel("Count")
    plt.show()
```

## 📚 Related Documentation

- **[Training Guide](training.md)** - How to train the model
- **[Model Architecture](architecture.md)** - Model design details
- **[Live Recognition](live-recognition.md)** - Real-time recognition setup
- **[Installation Guide](installation.md)** - Setup instructions

---

**Ready to prepare your ASL dataset! 📊** 