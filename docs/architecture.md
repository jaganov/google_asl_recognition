# 🏗️ Архитектура модели ASL Recognition

Детальное описание архитектуры модели для распознавания жестов ASL, основанной на решении победителя конкурса Google ASL Signs.

## 🎯 Обзор архитектуры

Модель использует гибридную архитектуру, сочетающую современные подходы к обработке временных последовательностей:

- **TCN (Temporal Convolutional Networks)** - для локальных временных паттернов
- **LSTM (Long Short-Term Memory)** - для долгосрочных зависимостей
- **Transformer** - для внимания и глобальных связей
- **Adaptive Regularization** - для борьбы с переобучением

## 🏗️ Полная архитектура

```
Input: (batch_size, seq_len, 543, 3)
    ↓
Preprocessing Layer
    ↓
Stem: Linear(744, 192) + BatchNorm + AdaptiveDropout
    ↓
TCN Block 1 (dilation=1, kernel=17)
    ↓
TCN Block 2 (dilation=2, kernel=17)
    ↓
TCN Block 3 (dilation=4, kernel=17)
    ↓
Bidirectional LSTM (2 layers, hidden_dim=96)
    ↓
Temporal Attention (8 heads)
    ↓
Conv1D Block 1 (kernel=17)
    ↓
Conv1D Block 2 (kernel=17)
    ↓
Conv1D Block 3 (kernel=17)
    ↓
Transformer Block (8 heads, expand=2)
    ↓
Top Layer: Linear(192, 192) + BatchNorm + AdaptiveDropout
    ↓
Multi-scale Pooling:
    ├── Global Average Pooling
    ├── Global Max Pooling
    └── Attention Pooling (4 heads)
    ↓
Concatenation: (192*3 = 576)
    ↓
Classifier: Linear(576, 192) → BatchNorm → SiLU → Dropout → Linear(192, 25)
    ↓
Output: (batch_size, 25)
```

## 🔧 Детали компонентов

### 1. Preprocessing Layer

```python
class PreprocessingLayer(nn.Module):
    def __init__(self, max_len=384, point_landmarks=None):
        super().__init__()
        self.max_len = max_len
        
        # Выбор ключевых landmarks
        face_landmarks = [33, 133, 362, 263, 61, 291, 199, 419, 17, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        left_hand = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521]
        right_hand = [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]
        self.point_landmarks = face_landmarks + left_hand + right_hand
```

**Функции:**
- Нормализация относительно носа (landmark 17)
- Выбор ключевых landmarks (62 точки)
- Вычисление motion features
- Обрезка до max_len кадров

### 2. Motion Features

```python
def compute_motion_features(self, x):
    # velocity (lag1)
    dx = torch.zeros_like(x)
    dx[:, :-1] = x[:, 1:] - x[:, :-1]
    
    # acceleration (lag2)
    dx2 = torch.zeros_like(x)
    dx2[:, :-2] = x[:, 2:] - x[:, :-2]
    
    # relative motion
    relative_motion = torch.zeros_like(x)
    # ... вычисление относительного движения
    
    # temporal consistency
    temporal_consistency = torch.zeros_like(x)
    # ... проверка согласованности движения
    
    # motion magnitude
    motion_magnitude = torch.norm(dx, dim=-1, keepdim=True)
    
    # motion direction
    motion_direction = torch.atan2(dx[..., 1], dx[..., 0]).unsqueeze(-1)
    
    return dx, dx2, relative_motion, temporal_consistency, motion_magnitude, motion_direction
```

**Выход:** 744 features (62 landmarks × 2 coordinates × 6 motion features)

### 3. TCN Blocks

```python
class TemporalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=17, dilation=1, drop_rate=0.2):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Causal convolution with dilation
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=self.padding, 
                              dilation=dilation, groups=dim)
        self.conv2 = nn.Conv1d(dim, dim, 1)  # Pointwise
        
        # Gated activation
        self.gate_conv = nn.Conv1d(dim, dim, kernel_size, padding=self.padding, 
                                  dilation=dilation, groups=dim)
        self.gate_conv2 = nn.Conv1d(dim, dim, 1)
        
        # Normalization
        self.bn = nn.BatchNorm1d(dim, momentum=0.95)
        self.dropout = nn.Dropout(drop_rate)
```

**Особенности:**
- **Causal convolution** - сохраняет временной порядок
- **Dilation** - расширяет рецептивное поле (1, 2, 4)
- **Gated activation** - улучшает градиентный поток
- **Residual connection** - стабилизирует обучение

### 4. Bidirectional LSTM

```python
class BidirectionalLSTM(nn.Module):
    def __init__(self, dim, hidden_dim=None, num_layers=2, drop_rate=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=drop_rate if num_layers > 1 else 0
        )
        
        # Projection back to original dimension
        self.projection = nn.Linear(hidden_dim * 2, dim)
        self.dropout = nn.Dropout(drop_rate)
```

**Функции:**
- **Двунаправленная обработка** - захватывает контекст в обе стороны
- **Многослойность** - 2 слоя для сложных зависимостей
- **Projection** - возврат к исходной размерности

### 5. Temporal Attention

```python
class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, drop_rate=0.2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Temporal attention
        self.temporal_q = nn.Linear(dim, dim)
        self.temporal_k = nn.Linear(dim, dim)
        self.temporal_v = nn.Linear(dim, dim)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop_rate)
        
        # Temporal position encoding
        self.temporal_pos_enc = nn.Parameter(torch.randn(1, 1000, dim))
```

**Особенности:**
- **Multi-head attention** - 8 голов для разных аспектов
- **Positional encoding** - временная информация
- **Scaled dot-product** - стабильное внимание

### 6. Conv1D Blocks

```python
class Conv1DBlock(nn.Module):
    def __init__(self, dim, kernel_size=17, drop_rate=0.2):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Causal padding
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=self.padding, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        
        # BatchNorm + Swish
        self.bn = nn.BatchNorm1d(dim, momentum=0.95)
        self.dropout = nn.Dropout(drop_rate)
```

**Особенности:**
- **Depthwise convolution** - эффективная обработка
- **Causal padding** - сохранение временного порядка
- **Swish activation** - современная функция активации

### 7. Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, expand=2, drop_rate=0.2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop_rate)
        self.attention_norm = nn.BatchNorm1d(dim, momentum=0.95)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.SiLU(),  # Swish
            nn.Dropout(drop_rate),
            nn.Linear(dim * expand, dim),
            nn.Dropout(drop_rate)
        )
        self.ffn_norm = nn.BatchNorm1d(dim, momentum=0.95)
```

**Особенности:**
- **Self-attention** - глобальные связи
- **Feed-forward** - нелинейные преобразования
- **BatchNorm** - стабилизация обучения

### 8. Multi-scale Pooling

```python
# Global average pooling
global_avg = self.temporal_pool1(x_transposed).squeeze(-1)

# Global max pooling
global_max = self.temporal_pool2(x_transposed).squeeze(-1)

# Attention pooling
attn_out, _ = self.attention_pool(x_for_attn, x_for_attn, x_for_attn)
global_attn = torch.mean(attn_out, dim=1)

# Combine pooling results
x_pooled = torch.cat([global_avg, global_max, global_attn], dim=1)
```

**Преимущества:**
- **Разные уровни абстракции** - локальные и глобальные паттерны
- **Устойчивость** - к вариациям длительности
- **Информативность** - сохранение важной информации

## 🎯 Адаптивная регуляризация

### Adaptive Dropout

```python
class AdaptiveDropout(nn.Module):
    def __init__(self, initial_p=0.1, final_p=0.6, warmup_epochs=30):
        super().__init__()
        self.initial_p = initial_p
        self.final_p = final_p
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def forward(self, x):
        if self.training:
            # Gradual dropout increase
            if self.current_epoch < self.warmup_epochs:
                p = self.initial_p + (self.final_p - self.initial_p) * (self.current_epoch / self.warmup_epochs)
            else:
                p = self.final_p
            return F.dropout(x, p=p, training=True)
        return x
```

**Преимущества:**
- **Плавная активация** - вместо резкого включения
- **Адаптивность** - подстраивается под прогресс обучения
- **Стабильность** - предотвращает резкие изменения

## 📊 Параметры модели

### Общие параметры
- **Total Parameters**: 1,968,601
- **Trainable Parameters**: 1,968,601
- **Input Dimension**: 744 (после preprocessing)
- **Hidden Dimension**: 192
- **Number of Classes**: 25
- **Max Sequence Length**: 384

### Параметры по компонентам
- **TCN Blocks**: ~300K параметров
- **LSTM**: ~150K параметров
- **Attention**: ~200K параметров
- **Conv1D**: ~400K параметров
- **Transformer**: ~300K параметров
- **Classifier**: ~600K параметров

## 🔍 Ключевые особенности

### 1. Временное моделирование
- **TCN** - локальные паттерны
- **LSTM** - долгосрочные зависимости
- **Attention** - глобальные связи

### 2. Эффективность
- **Depthwise convolutions** - уменьшение параметров
- **Causal padding** - сохранение порядка
- **Residual connections** - стабилизация градиентов

### 3. Регуляризация
- **Adaptive Dropout** - плавная активация
- **BatchNorm** - нормализация
- **Label Smoothing** - улучшение обобщения

### 4. Масштабируемость
- **Modular design** - легко изменять компоненты
- **Configurable** - настраиваемые параметры
- **Extensible** - возможность добавления новых блоков

## 🎯 Преимущества архитектуры

1. **Комплексное моделирование** - сочетание разных подходов
2. **Эффективность** - оптимальное использование параметров
3. **Стабильность** - устойчивое обучение
4. **Интерпретируемость** - внимание показывает важные кадры
5. **Масштабируемость** - легко адаптировать под новые задачи

## 📚 Дополнительные ресурсы

- [Temporal Convolutional Networks](https://arxiv.org/abs/1609.03499)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LSTM Networks](https://arxiv.org/abs/1503.04069)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)

---

**Архитектура обеспечивает оптимальный баланс между сложностью и эффективностью! 🏗️** 