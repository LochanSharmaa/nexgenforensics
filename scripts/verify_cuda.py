import sys
print("Python:", sys.version)
print("Python Executable:", sys.executable)

try:
    import torch
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Count:", torch.cuda.device_count())
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
        print("Total Memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
        x = torch.randn(100, 100, device="cuda")
        print("CUDA Tensor allocation test PASSED. Sum:", float(x.sum().item()))
    else:
        print("WARNING: CUDA is NOT available!")
except Exception as e:
    print("Error importing/testing PyTorch:", e)

for pkg in ["timm", "insightface", "mxnet", "cv2", "onnxruntime", "sklearn", "scipy"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"Package '{pkg}': {ver}")
    except ImportError:
        print(f"Package '{pkg}': NOT INSTALLED")
