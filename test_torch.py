import sys
print("Python:", sys.version)
print("Python executable:", sys.executable)
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory (bytes):", torch.cuda.get_device_properties(0).total_memory)
    x = torch.randn(3, 3).cuda()
    print("GPU tensor sum:", x.sum().item())
    print("Peak GPU memory MB:", torch.cuda.max_memory_allocated(0) / 1024**2)
print("All checks passed.")
