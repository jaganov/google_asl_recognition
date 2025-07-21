#!/usr/bin/env python3
"""
Тестовый скрипт для проверки загрузки модели ASL
"""

import torch
import json
from pathlib import Path
import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def test_model_loading():
    """Тестирует загрузку модели"""
    print("🧪 Тестирование загрузки модели ASL")
    print("=" * 50)
    
    # Путь к модели
    model_path = "models/asl_model_v20250720_080209.pth"
    
    # Проверяем существование файла
    if not Path(model_path).exists():
        print(f"❌ Файл модели не найден: {model_path}")
        return False
    
    print(f"✅ Файл модели найден: {model_path}")
    
    # Загружаем маппинг знаков
    sign_mapping_path = "dataset25/sign_to_prediction_index_map.json"
    if Path(sign_mapping_path).exists():
        with open(sign_mapping_path, 'r') as f:
            sign_mapping = json.load(f)
        classes = [""] * len(sign_mapping)
        for sign, idx in sign_mapping.items():
            classes[idx] = sign
        print(f"✅ Маппинг знаков загружен: {len(classes)} классов")
        print(f"   Классы: {classes}")
    else:
        print(f"⚠️ Маппинг знаков не найден: {sign_mapping_path}")
        classes = [f"class_{i}" for i in range(25)]
    
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Устройство: {device}")
    
    try:
        # Загружаем checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        print(f"✅ Checkpoint загружен")
        print(f"   Ключи в state_dict: {len(state_dict)}")
        
        # Анализируем структуру
        model_keys = list(state_dict.keys())
        print(f"   Первые 10 ключей: {model_keys[:10]}")
        
        # Определяем размерность входа
        input_dim = 744  # Из манифеста
        num_classes = len(classes)
        
        print(f"   Размерность входа: {input_dim}")
        print(f"   Количество классов: {num_classes}")
        
        # Создаем модель
        class TestASLModel(torch.nn.Module):
            def __init__(self, input_dim, num_classes, max_len=384, dim=192):
                super().__init__()
                self.max_len = max_len
                self.dim = dim
                
                # Preprocessing
                self.preprocessing = PreprocessingLayer(max_len)
                
                # Stem with improved initialization
                self.stem_conv = torch.nn.Linear(input_dim, dim, bias=False)
                self.stem_bn = torch.nn.BatchNorm1d(dim, momentum=0.95)
                self.stem_dropout = AdaptiveDropout(initial_p=0.1, final_p=0.3, warmup_epochs=20)
                
                # Temporal Convolutional Network (TCN) - 3 blocks with different dilations
                self.tcn1 = TemporalConvBlock(dim, kernel_size=17, dilation=1, drop_rate=0.15)
                self.tcn2 = TemporalConvBlock(dim, kernel_size=17, dilation=2, drop_rate=0.2)
                self.tcn3 = TemporalConvBlock(dim, kernel_size=17, dilation=4, drop_rate=0.25)
                
                # Projection for attention pooling
                self.attention_projection = torch.nn.Linear(dim, dim)
                
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
                self.top_conv = torch.nn.Linear(dim, dim)
                self.top_bn = torch.nn.BatchNorm1d(dim, momentum=0.95)
                self.adaptive_dropout = AdaptiveDropout(initial_p=0.2, final_p=0.5, warmup_epochs=25)
                
                # Improved pooling - add attention pooling
                self.temporal_pool1 = torch.nn.AdaptiveAvgPool1d(1)
                self.temporal_pool2 = torch.nn.AdaptiveMaxPool1d(1)
                self.attention_pool = torch.nn.MultiheadAttention(dim, num_heads=4, batch_first=True, dropout=0.1)
                
                # Final classifier with improved architecture
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(dim * 3, dim),  # dim*3 for avg + max + attention pooling
                    torch.nn.BatchNorm1d(dim),
                    torch.nn.SiLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(dim, num_classes)
                )
            
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
        
        model = TestASLModel(input_dim, num_classes).to(device)
        
        # Загружаем веса
        model.load_state_dict(state_dict)
        print("✅ Веса модели загружены успешно")
        
        # Переводим в режим оценки
        model.eval()
        
        # Тестируем предсказание на случайных данных
        print("\n🧪 Тестирование предсказания...")
        
        # Создаем тестовые данные (batch=1, seq=16, landmarks=543, coords=3)
        test_data = torch.randn(1, 16, 543, 3).to(device)
        
        with torch.no_grad():
            output = model(test_data)
            probabilities = torch.softmax(output, dim=1)
            
            # Получаем топ-3 предсказания
            top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
            
            print("✅ Предсказание выполнено успешно")
            print("   Топ-3 предсказания:")
            for i in range(3):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                sign_name = classes[idx] if idx < len(classes) else f"Unknown_{idx}"
                print(f"   {i+1}. {sign_name}: {prob:.3f}")
        
        print("\n🎉 Все тесты пройдены успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✅ Модель готова к использованию в live recognition!")
    else:
        print("\n❌ Есть проблемы с загрузкой модели") 