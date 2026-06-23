# Training Backend

This project now uses one supported backend:

```text
native_windows_torch_cuda
```

It trains directly on the laptop GPU through PyTorch CUDA on Windows.

```text
GPU: NVIDIA GeForce RTX 5050 Laptop GPU
VRAM: 8 GB
Training outputs: E:\ResUpNet\runs
Runtime cache:    E:\ResUpNet\cache
Temporary files:  E:\ResUpNet\tmp
```

The repository does not use alternate deep-learning backends. Keep the
environment PyTorch-only so GPU setup, checkpoint format, and evaluation remain
consistent.

## Install Backend

Activate the E-drive venv:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\activate_resupnet.ps1
```

Install PyTorch CUDA:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_brats.txt
```

Verify GPU visibility:

```powershell
python -B -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))"
```

Expected result:

```text
True
NVIDIA GeForce RTX 5050 Laptop GPU
about 8.0 GB
```
