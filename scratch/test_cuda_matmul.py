import torch

print("PyTorch Version:", torch.__version__)
print("PyTorch CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print("Device Name:", device_name)
    try:
        device = torch.device("cuda")
        x = torch.randn(32, 64, device=device)
        y = torch.randn(64, 64, device=device)
        z = x @ y
        print("CUDA matmul OK, shape:", z.shape)
    except Exception as e:
        print("CUDA matmul FAILED with error:", e)
else:
    print("No CUDA device available.")
