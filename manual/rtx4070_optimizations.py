"""
Оптимизации для RTX4070 и дополнительные улучшения модели ASL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Optional, Tuple

# ============================================================================
# ОПТИМИЗАЦИИ ДЛЯ RTX4070
# ============================================================================

def setup_rtx4070_optimizations():
    """
    Настройка оптимизаций для RTX4070
    """
    print("🔧 Настройка оптимизаций для RTX4070...")
    
    # Включаем TF32 для ускорения на Ampere архитектуре
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Оптимизации cuDNN
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Устанавливаем оптимальный алгоритм для convolutions
    torch.backends.cudnn.enabled = True
    
    print("   ✅ TF32 включен")
    print("   ✅ cuDNN benchmark включен")
    print("   ✅ Оптимизации применены")

class MixedPrecisionTrainer:
    """
    Тренер с поддержкой mixed precision для ускорения на RTX4070
    """
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()
    
    def train_step(self, sequences, labels, criterion):
        """
        Один шаг тренировки с mixed precision
        """
        self.optimizer.zero_grad()
        
        # Forward pass с autocast
        with autocast():
            outputs = self.model(sequences)
            loss = criterion(outputs, labels)
        
        # Backward pass с scaler
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item(), outputs

# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ УЛУЧШЕНИЯ МОДЕЛИ
# ============================================================================

class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) для регуляризации
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ImprovedConv1DBlock(nn.Module):
    """
    Улучшенный 1D CNN блок с DropPath
    """
    def __init__(self, dim, kernel_size=17, drop_rate=0.2, drop_path_rate=0.2):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=self.padding, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        
        # BatchNorm + Swish
        self.bn = nn.BatchNorm1d(dim, momentum=0.95)
        self.dropout = nn.Dropout(drop_rate)
        self.drop_path = DropPath(drop_path_rate)
        
        # Residual connection
        self.residual = nn.Conv1d(dim, dim, 1) if dim != dim else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x.transpose(1, 2))
        
        # Causal convolution
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # Применяем causal padding
        x = x[:, :, :-self.padding] if self.padding > 0 else x
        
        x = self.bn(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        
        x = x.transpose(1, 2)
        x = x + residual.transpose(1, 2)
        
        return x

class AWP(nn.Module):
    """
    Adversarial Weight Perturbation для регуляризации
    """
    def __init__(self, model, epsilon=0.2, alpha=0.2):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, x, y, criterion, optimizer):
        """
        Атака на веса модели
        """
        self._save()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                eps = self.epsilon * param.abs().detach()
                param.data.add_(eps)
                self.backup_eps[name] = eps.data.clone()
        
        optimizer.zero_grad()
        outputs = self.model(x)
        loss = criterion(outputs, y)
        loss.backward()
        
        self._restore()
        return loss

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

    def _restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

class ImprovedASLModel(nn.Module):
    """
    Улучшенная модель ASL с дополнительными оптимизациями
    """
    def __init__(self, input_dim, num_classes, max_len=384, dim=192, dropout_step=0):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        
        # Preprocessing
        self.preprocessing = PreprocessingLayer(max_len)
        
        # Stem
        self.stem_conv = nn.Linear(input_dim, dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim, momentum=0.95)
        
        # Улучшенные 1D CNN + Transformer blocks
        ksize = 17
        
        # Первая группа
        self.conv1 = ImprovedConv1DBlock(dim, ksize, drop_rate=0.2, drop_path_rate=0.2)
        self.conv2 = ImprovedConv1DBlock(dim, ksize, drop_rate=0.2, drop_path_rate=0.2)
        self.conv3 = ImprovedConv1DBlock(dim, ksize, drop_rate=0.2, drop_path_rate=0.2)
        self.transformer1 = TransformerBlock(dim, expand=2)
        
        # Вторая группа
        self.conv4 = ImprovedConv1DBlock(dim, ksize, drop_rate=0.2, drop_path_rate=0.2)
        self.conv5 = ImprovedConv1DBlock(dim, ksize, drop_rate=0.2, drop_path_rate=0.2)
        self.conv6 = ImprovedConv1DBlock(dim, ksize, drop_rate=0.2, drop_path_rate=0.2)
        self.transformer2 = TransformerBlock(dim, expand=2)
        
        # Top layers
        self.top_conv = nn.Linear(dim, dim * 2)
        self.late_dropout = LateDropout(0.8, start_step=dropout_step)
        self.classifier = nn.Linear(dim * 2, num_classes)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, seq, landmarks, 3)
        
        # Preprocessing
        x = self.preprocessing(x)
        
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x.transpose(1, 2)).transpose(1, 2)
        
        # 1D CNN + Transformer blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.transformer1(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.transformer2(x)
        
        # Top layers
        x = self.top_conv(x)
        x = torch.mean(x, dim=1)
        x = self.late_dropout(x)
        x = self.classifier(x)
        
        return x

# ============================================================================
# ОПТИМИЗИРОВАННЫЙ ТРЕНИРОВЩИК
# ============================================================================

class OptimizedTrainer:
    """
    Оптимизированный тренер для RTX4070
    """
    def __init__(self, model, device, use_mixed_precision=True):
        self.model = model
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        if use_mixed_precision:
            self.scaler = GradScaler()
        
        # AWP
        self.awp = AWP(model, epsilon=0.2, alpha=0.2)
    
    def train_step(self, sequences, labels, criterion, optimizer, epoch):
        """
        Один шаг тренировки с оптимизациями
        """
        sequences = sequences.to(self.device)
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if self.use_mixed_precision:
            with autocast():
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
            
            # Backward pass с scaler
            self.scaler.scale(loss).backward()
            
            # AWP после 15 эпохи
            if epoch >= 15:
                self.scaler.unscale_(optimizer)
                self.awp.attack_backward(sequences, labels, criterion, optimizer)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = self.model(sequences)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # AWP после 15 эпохи
            if epoch >= 15:
                self.awp.attack_backward(sequences, labels, criterion, optimizer)
            
            optimizer.step()
        
        return loss.item(), outputs

# ============================================================================
# УТИЛИТЫ ДЛЯ ОПТИМИЗАЦИИ ПАМЯТИ
# ============================================================================

def optimize_memory_usage():
    """
    Оптимизация использования памяти GPU
    """
    print("💾 Оптимизация использования памяти...")
    
    # Очищаем кэш CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Устанавливаем оптимальные настройки памяти
    torch.cuda.set_per_process_memory_fraction(0.9)  # Используем 90% памяти
    
    print("   ✅ Кэш CUDA очищен")
    print("   ✅ Настройки памяти оптимизированы")

def get_optimal_batch_size(model, input_shape, device, max_memory_gb=12):
    """
    Автоматический подбор оптимального размера батча для RTX4070
    """
    print(f"🔍 Подбор оптимального размера батча...")
    
    # Начинаем с небольшого размера
    batch_size = 8
    max_batch_size = 64
    
    while batch_size <= max_batch_size:
        try:
            # Тестируем с текущим размером батча
            test_input = torch.randn(batch_size, *input_shape, device=device)
            
            # Очищаем память
            torch.cuda.empty_cache()
            
            # Пробуем forward pass
            with torch.no_grad():
                _ = model(test_input)
            
            # Если успешно, увеличиваем размер
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Если не хватает памяти, возвращаем предыдущий размер
                optimal_batch_size = batch_size // 2
                print(f"   ✅ Оптимальный размер батча: {optimal_batch_size}")
                return optimal_batch_size
            else:
                raise e
    
    print(f"   ✅ Оптимальный размер батча: {batch_size // 2}")
    return batch_size // 2

# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ АУГМЕНТАЦИИ
# ============================================================================

class AdvancedAugmentation:
    """
    Расширенные аугментации на основе решения победителя
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def temporal_cutout(self, x, cutout_ratio=0.1):
        """
        Временной cutout - маскирование случайных временных интервалов
        """
        if random.random() > self.p:
            return x
        
        seq_len = x.shape[0]
        cutout_len = int(seq_len * cutout_ratio)
        
        if cutout_len > 0:
            start_idx = random.randint(0, seq_len - cutout_len)
            x[start_idx:start_idx + cutout_len] = 0
        
        return x
    
    def spatial_cutout(self, x, cutout_ratio=0.1):
        """
        Пространственный cutout - маскирование случайных landmarks
        """
        if random.random() > self.p:
            return x
        
        num_landmarks = x.shape[1]
        cutout_count = int(num_landmarks * cutout_ratio)
        
        if cutout_count > 0:
            cutout_indices = random.sample(range(num_landmarks), cutout_count)
            x[:, cutout_indices] = 0
        
        return x
    
    def random_noise(self, x, noise_std=0.01):
        """
        Добавление случайного шума
        """
        if random.random() > self.p:
            return x
        
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    def apply_all_augmentations(self, x):
        """
        Применение всех аугментаций
        """
        x = self.temporal_resample(x)
        x = self.random_masking(x)
        x = self.horizontal_flip(x)
        x = self.random_affine(x)
        x = self.temporal_cutout(x)
        x = self.spatial_cutout(x)
        x = self.random_noise(x)
        
        return x

# ============================================================================
# ФУНКЦИИ ДЛЯ АНАЛИЗА ПРОИЗВОДИТЕЛЬНОСТИ
# ============================================================================

def benchmark_model(model, input_shape, device, num_runs=100):
    """
    Бенчмарк производительности модели
    """
    print(f"⚡ Бенчмарк производительности модели...")
    
    model.eval()
    input_tensor = torch.randn(1, *input_shape, device=device)
    
    # Разогрев
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Синхронизируем GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Измеряем время
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    end_time.record()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    avg_time = start_time.elapsed_time(end_time) / num_runs
    fps = 1000 / avg_time  # FPS
    
    print(f"   ⏱️  Среднее время inference: {avg_time:.2f} ms")
    print(f"   🎯 FPS: {fps:.1f}")
    
    return avg_time, fps

def monitor_gpu_usage():
    """
    Мониторинг использования GPU
    """
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"💾 GPU Memory:")
        print(f"   Выделено: {memory_allocated:.2f} GB")
        print(f"   Зарезервировано: {memory_reserved:.2f} GB")
        print(f"   Всего: {memory_total:.2f} GB")
        print(f"   Использование: {memory_allocated/memory_total*100:.1f}%")

# ============================================================================
# ЭКСПОРТ МОДЕЛИ
# ============================================================================

def export_model_for_inference(model, input_shape, output_path="models/asl_model_optimized.pt"):
    """
    Экспорт модели для быстрого inference
    """
    print(f"📦 Экспорт модели для inference...")
    
    model.eval()
    
    # Создаем пример входных данных
    example_input = torch.randn(1, *input_shape)
    
    # Экспортируем с TorchScript
    try:
        traced_model = torch.jit.trace(model, example_input)
        torch.jit.save(traced_model, output_path)
        print(f"   ✅ Модель экспортирована в: {output_path}")
        
        # Тестируем экспортированную модель
        loaded_model = torch.jit.load(output_path)
        with torch.no_grad():
            output = loaded_model(example_input)
        print(f"   ✅ Экспортированная модель работает корректно")
        
    except Exception as e:
        print(f"   ⚠️ Ошибка при экспорте: {e}")
        print(f"   💾 Сохраняем обычную модель...")
        torch.save(model.state_dict(), output_path.replace('.pt', '_state_dict.pth'))

if __name__ == "__main__":
    print("🔧 RTX4070 Оптимизации и улучшения модели ASL")
    print("=" * 60)
    
    # Настройка оптимизаций
    setup_rtx4070_optimizations()
    optimize_memory_usage()
    
    print("\n✅ Все оптимизации применены!")
    print("💡 Используйте эти функции в основном коде тренировки") 