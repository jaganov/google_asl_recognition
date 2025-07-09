# verify_py312.py - Python 3.12 compatibility check
import sys
import importlib.util

def check_python312_compatibility():
    """Check if all packages work with Python 3.12"""
    
    print(f"🐍 Python Version: {sys.version}")
    
    # Check if Python 3.12
    if sys.version_info[:2] != (3, 12):
        print(f"⚠️  Warning: Expected Python 3.12, got {sys.version_info[:2]}")
    else:
        print("✅ Python 3.12 confirmed")
    
    # Critical packages for ASL project
    critical_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'transformers': 'Transformers', 
        'timm': 'Timm Vision Models',
        'cv2': 'OpenCV',
        'decord': 'Decord Video Loader',
        'gradio': 'Gradio Demo',
        'datasets': 'Hugging Face Datasets',
        'tensorflow_datasets': 'TensorFlow Datasets'
    }
    
    print("\n🔍 Checking critical packages...")
    failed = []
    
    for module, name in critical_packages.items():
        try:
            spec = importlib.util.find_spec(module)
            if spec is not None:
                mod = importlib.import_module(module)
                version = getattr(mod, '__version__', 'unknown')
                print(f"✅ {name}: {version}")
            else:
                print(f"❌ {name}: Not found")
                failed.append(module)
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            failed.append(module)
    
    # Check CUDA with PyTorch
    try:
        import torch
        print(f"\n🖥️  PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️  CUDA not available")
    except ImportError:
        print("❌ PyTorch not installed")
        failed.append('torch')
    
    # Test dataset access
    try:
        import os
        dataset_path = '../data/google_asl_signs'
        if os.path.exists(dataset_path):
            size_gb = sum(os.path.getsize(os.path.join(dp, f)) 
                         for dp, dn, fn in os.walk(dataset_path) 
                         for f in fn) / 1e9
            print(f"\n📚 Dataset: {size_gb:.1f} GB found at {dataset_path}")
        else:
            print(f"\n⚠️  Dataset not found at {dataset_path}")
    except Exception as e:
        print(f"\n❌ Dataset check failed: {e}")
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: uv pip install " + " ".join(failed))
        return False
    else:
        print("\n🎉 All packages ready for Google ASL project!")
        return True

if __name__ == "__main__":
    check_python312_compatibility()