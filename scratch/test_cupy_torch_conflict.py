print("Importing PyTorch...")
import torch
device = torch.device("cuda")
print("CUDA Available in PyTorch:", torch.cuda.is_available())

print("Initializing PyTorch CUDA...")
# Force PyTorch to initialize cuBLAS context
a = torch.randn(32, 64, device=device)
b = torch.randn(64, 64, device=device)
c = a @ b
print("PyTorch CUDA matmul succeeded, shape:", c.shape)

print("Importing and initializing CuPy...")
import cupy as cp
import numpy as np
x_cp = cp.asarray(np.random.randn(100, 100))
y_cp = x_cp * 2
print("CuPy VRAM Info:", cp.cuda.Device(0).mem_info)
print("CuPy GPU operation succeeded!")

print("Running PyTorch CUDA matmul again...")
d = a @ b
print("PyTorch CUDA matmul 2 succeeded, shape:", d.shape)
