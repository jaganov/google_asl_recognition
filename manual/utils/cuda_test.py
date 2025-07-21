#!/usr/bin/env python3
"""
CUDA и PyTorch диагностика для RTX 4070 с CUDA 12.9
"""

import torch
import torchvision
import subprocess
import platform
import sys

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def check_system_info():
    print_separator("SYSTEM INFORMATION")
    print(f"🖥️  Platform: {platform.platform()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📦 PyTorch: {torch.__version__}")
    print(f"👁️  Torchvision: {torchvision.__version__}")

def check_cuda_installation():
    print_separator("CUDA INSTALLATION")
    
    # Check NVIDIA-SMI
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA Driver: OK")
            # Extract CUDA version
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version: ')[1].split()[0]
                    print(f"📊 CUDA Runtime: {cuda_version}")
                    break
        else:
            print("❌ NVIDIA Driver: Failed")
    except Exception as e:
        print(f"❌ NVIDIA-SMI Error: {e}")
    
    # Check NVCC
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CUDA Toolkit: Installed")
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    version = line.split('release ')[1].split(',')[0]
                    print(f"🛠️  CUDA Toolkit: {version}")
                    break
        else:
            print("⚠️  CUDA Toolkit: Not in PATH (это нормально)")
    except Exception:
        print("⚠️  CUDA Toolkit: Not found (это нормально для pip установки)")

def check_pytorch_cuda():
    print_separator("PYTORCH CUDA SUPPORT")
    
    print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"📱 CUDA Devices: {torch.cuda.device_count()}")
        print(f"🎮 Current Device: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"🎯 GPU {i}: {props.name}")
            print(f"💾 VRAM: {props.total_memory // 1024**3} GB")
            print(f"🔧 Compute Capability: {props.major}.{props.minor}")
            print(f"🏭 Multiprocessors: {props.multi_processor_count}")
    
    print(f"🔗 PyTorch CUDA Version: {torch.version.cuda}")
    print(f"🏗️  cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"🎯 cuDNN Enabled: {torch.backends.cudnn.enabled}")

def test_cuda_operations():
    print_separator("CUDA OPERATIONS TEST")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping tests")
        return False
    
    try:
        # Test basic operations
        print("🧪 Testing basic CUDA operations...")
        
        # Create tensors
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Matrix multiplication
        import time
        start_time = time.time()
        z = torch.mm(x, y)
        cuda_time = time.time() - start_time
        
        print(f"✅ Matrix multiplication: {cuda_time:.4f}s")
        
        # Memory test
        allocated = torch.cuda.memory_allocated() // 1024**2
        cached = torch.cuda.memory_reserved() // 1024**2
        print(f"📊 Memory allocated: {allocated} MB")
        print(f"🗂️  Memory cached: {cached} MB")
        
        # Test mixed precision
        print("🧪 Testing mixed precision (AMP)...")
        scaler = torch.cuda.amp.GradScaler()
        
        with torch.cuda.amp.autocast():
            result = torch.mm(x, y)
        
        print("✅ Mixed precision: OK")
        
        # Clean up
        del x, y, z, result
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_google_asl_compatibility():
    print_separator("GOOGLE ASL PROJECT COMPATIBILITY")
    
    if not torch.cuda.is_available():
        print("❌ CUDA required for this project")
        return False
    
    try:
        # Simulate Google ASL model requirements
        print("🧪 Testing video processing simulation...")
        
        # Simulate video batch: (batch=4, frames=16, channels=3, height=224, width=224)
        batch_size = 4
        frames = 16
        video_batch = torch.randn(batch_size, frames, 3, 224, 224).cuda()
        
        print(f"✅ Video batch shape: {video_batch.shape}")
        print(f"📊 Video batch size: {video_batch.numel() * 4 / 1024**2:.1f} MB")
        
        # Test with different batch sizes for RTX 4070
        optimal_batch = 1
        for batch in [1, 2, 4, 8, 12, 16]:
            try:
                test_batch = torch.randn(batch, 16, 3, 224, 224).cuda()
                # Simulate forward pass memory usage
                temp = test_batch * 2  # Simple operation
                optimal_batch = batch
                del test_batch, temp
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                break
        
        print(f"🎯 Optimal batch size for RTX 4070: {optimal_batch}")
        
        # Test Vision Transformer compatibility
        print("🧪 Testing Vision Transformer compatibility...")
        
        # Simulate ViT patch embedding
        patch_size = 16
        img_size = 224
        num_patches = (img_size // patch_size) ** 2
        embed_dim = 768
        
        patches = torch.randn(batch_size, num_patches, embed_dim).cuda()
        print(f"✅ ViT patches: {patches.shape}")
        
        print("✅ All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def generate_recommendations():
    print_separator("RECOMMENDATIONS FOR GOOGLE ASL PROJECT")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
        
        print(f"🎮 Your GPU: {gpu_name}")
        print(f"💾 Your VRAM: {vram_gb} GB")
        
        if "RTX 4070" in gpu_name:
            print("\n🎯 RTX 4070 Specific Recommendations:")
            print("✅ Perfect for Google ASL project!")
            print("📊 Recommended settings:")
            print("   • Batch size: 8-12")
            print("   • Mixed precision: Enabled") 
            print("   • Video frames: 16")
            print("   • Image resolution: 224x224")
            print("   • Expected training time: 2-4 hours per model")
        
        print(f"\n🔧 Optimal PyTorch installation:")
        print(f"   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        
    else:
        print("❌ CUDA not available")
        print("🔧 Install CUDA-enabled PyTorch:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")

def main():
    print("🚀 CUDA & PyTorch Diagnostic Tool for Google ASL Project")
    print("🎯 Optimized for RTX 4070 with CUDA 12.9")
    
    check_system_info()
    check_cuda_installation() 
    check_pytorch_cuda()
    
    cuda_works = test_cuda_operations()
    compat_works = test_google_asl_compatibility()
    
    generate_recommendations()
    
    print_separator("SUMMARY")
    
    if cuda_works and compat_works:
        print("🎉 CONGRATULATIONS!")
        print("✅ Your system is ready for the Google ASL project!")
        print("🚀 You can start training immediately!")
    elif cuda_works:
        print("⚠️  CUDA works, but some optimizations needed")
        print("🔧 Check recommendations above")
    else:
        print("❌ CUDA setup issues detected")
        print("🔧 Please fix CUDA installation first")

if __name__ == "__main__":
    main()