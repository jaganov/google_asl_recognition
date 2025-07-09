# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Conv1DBlock(nn.Module):
    """1D CNN блок (как у победителя)"""
    
    def __init__(self, dim: int, ksize: int = 17, drop_rate: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=ksize, padding=ksize//2, groups=dim)
        self.bn = nn.BatchNorm1d(dim, momentum=0.95)
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x):
        # x: (batch, seq_len, dim) -> (batch, dim, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = F.silu(x)  # Swish activation
        x = self.dropout(x)
        # (batch, dim, seq_len) -> (batch, seq_len, dim)
        return x.transpose(1, 2)

class TransformerBlock(nn.Module):
    """Transformer блок (как у победителя)"""
    
    def __init__(self, dim: int, expand: int = 2, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.expand = expand
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * expand, dim),
            nn.Dropout(0.1)
        )
        
        # Layer norms
        self.norm1 = nn.BatchNorm1d(dim, momentum=0.95)
        self.norm2 = nn.BatchNorm1d(dim, momentum=0.95)
        
    def forward(self, x):
        # x: (batch, seq_len, dim)
        
        # Self-attention
        residual = x
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        x = residual + attn_out
        
        # Feed forward
        residual = x
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = residual + self.ffn(x)
        
        return x

class ASL1DCNNTransformer(nn.Module):
    """1D CNN + Transformer модель (как у победителя)"""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int = 250,
                 max_len: int = 384,
                 dim: int = 192,
                 dropout_step: int = 0):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        
        # Input projection
        self.stem_conv = nn.Linear(input_dim, dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim, momentum=0.95)
        
        # 1D CNN + Transformer blocks (как у победителя)
        ksize = 17
        
        # Первая группа: 3 Conv1D + 1 Transformer
        self.block1_conv1 = Conv1DBlock(dim, ksize, drop_rate=0.2)
        self.block1_conv2 = Conv1DBlock(dim, ksize, drop_rate=0.2)
        self.block1_conv3 = Conv1DBlock(dim, ksize, drop_rate=0.2)
        self.block1_transformer = TransformerBlock(dim, expand=2)
        
        # Вторая группа: 3 Conv1D + 1 Transformer
        self.block2_conv1 = Conv1DBlock(dim, ksize, drop_rate=0.2)
        self.block2_conv2 = Conv1DBlock(dim, ksize, drop_rate=0.2)
        self.block2_conv3 = Conv1DBlock(dim, ksize, drop_rate=0.2)
        self.block2_transformer = TransformerBlock(dim, expand=2)
        
        # Дополнительные блоки для больших моделей (dim=384)
        if dim == 384:
            # Третья группа
            self.block3_conv1 = Conv1DBlock(dim, ksize, drop_rate=0.2)
            self.block3_conv2 = Conv1DBlock(dim, ksize, drop_rate=0.2)
            self.block3_conv3 = Conv1DBlock(dim, ksize, drop_rate=0.2)
            self.block3_transformer = TransformerBlock(dim, expand=2)
            
            # Четвертая группа
            self.block4_conv1 = Conv1DBlock(dim, ksize, drop_rate=0.2)
            self.block4_conv2 = Conv1DBlock(dim, ksize, drop_rate=0.2)
            self.block4_conv3 = Conv1DBlock(dim, ksize, drop_rate=0.2)
            self.block4_transformer = TransformerBlock(dim, expand=2)
        
        # Top layers
        self.top_conv = nn.Linear(dim, dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.8)
        self.dropout_step = dropout_step
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim) - уже препроцессированные фичи
        
        # Обрезка по времени
        if x.shape[1] > self.max_len:
            x = x[:, :self.max_len]
        
        # Input projection
        x = self.stem_conv(x)
        x = self.stem_bn(x.transpose(1, 2)).transpose(1, 2)
        
        # Первая группа блоков
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = self.block1_conv3(x)
        x = self.block1_transformer(x)
        
        # Вторая группа блоков
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_conv3(x)
        x = self.block2_transformer(x)
        
        # Дополнительные блоки для больших моделей
        if self.dim == 384:
            x = self.block3_conv1(x)
            x = self.block3_conv2(x)
            x = self.block3_conv3(x)
            x = self.block3_transformer(x)
            
            x = self.block4_conv1(x)
            x = self.block4_conv2(x)
            x = self.block4_conv3(x)
            x = self.block4_transformer(x)
        
        # Top layers
        x = self.top_conv(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, dim*2)
        
        # Late dropout (как у победителя)
        x = self.dropout(x)
        
        # Classifier
        x = self.classifier(x)
        
        return x

class ASLEnsemble(nn.Module):
    """Ансамбль моделей (как у победителя)"""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0] * len(models)
        else:
            self.weights = weights
            
        # Нормализуем веса
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def forward(self, x):
        outputs = []
        
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(output * weight)
        
        # Взвешенная сумма
        ensemble_output = sum(outputs)
        
        return ensemble_output

def get_model(input_dim: int, 
              num_classes: int = 250, 
              max_len: int = 384, 
              dim: int = 192,
              dropout_step: int = 0) -> ASL1DCNNTransformer:
    """
    Создает модель (как у победителя)
    
    Args:
        input_dim: Размерность входных фич
        num_classes: Количество классов
        max_len: Максимальная длина последовательности
        dim: Размерность модели (192 или 384)
        dropout_step: Шаг для late dropout
    
    Returns:
        ASL1DCNNTransformer: Модель
    """
    model = ASL1DCNNTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        max_len=max_len,
        dim=dim,
        dropout_step=dropout_step
    )
    
    print(f"🎯 Создана модель (как у победителя):")
    print(f"   Архитектура: 1D CNN + Transformer")
    print(f"   Размерность: {dim}")
    print(f"   Максимальная длина: {max_len}")
    print(f"   Количество классов: {num_classes}")
    print(f"   Параметры: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def test_model():
    """Тест модели"""
    print("🧪 Тестируем модель...")
    
    # Создаем модель
    input_dim = 543 * 6  # Пример: 543 точки * 6 фич (x, y, dx, dy, dx2, dy2)
    model = get_model(input_dim=input_dim, dim=192)
    
    # Тестовый батч
    batch_size = 4
    seq_len = 64
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Тест успешен!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return model

if __name__ == "__main__":
    test_model() 