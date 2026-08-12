#!/usr/bin/env python3
"""Fail-fast smoke test for the supported single RTX 5090 environment."""

import os
import sys
from importlib.metadata import version

import mmcv
import mmengine
import mmdet
import torch
import torchvision
from fairscale.nn.checkpoint import checkpoint_wrapper
from mmcv.ops import get_compiler_version, get_compiling_cuda_version, nms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


require(sys.version_info[:2] == (3, 10), f"Expected Python 3.10, got {sys.version}")
require(torch.__version__.startswith("2.7.1+cu128"), f"Unexpected torch: {torch.__version__}")
require(torchvision.__version__.startswith("0.22.1+cu128"),
        f"Unexpected torchvision: {torchvision.__version__}")
require(mmengine.__version__ == "0.10.7", f"Unexpected MMEngine: {mmengine.__version__}")
require(mmcv.__version__.startswith("2.2.0"), f"Unexpected MMCV: {mmcv.__version__}")
require(mmdet.__version__ == "3.3.0", f"Unexpected vendored MMDetection: {mmdet.__version__}")
require(version("fairscale") == "0.4.13", f"Unexpected FairScale: {version('fairscale')}")
require(callable(checkpoint_wrapper), "FairScale checkpoint_wrapper is unavailable")
require(torch.cuda.is_available(), "torch.cuda.is_available() is false")
require(torch.cuda.device_count() == 1,
        "Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0 when running this check")

device = torch.device("cuda:0")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
require(capability >= (12, 0), f"Expected Blackwell sm_120+, got {capability}")
require("sm_120" in torch.cuda.get_arch_list(),
        f"PyTorch wheel lacks sm_120: {torch.cuda.get_arch_list()}")

x = torch.randn(1024, 1024, device=device, requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
require(torch.isfinite(loss).item(), "CUDA matmul/backward produced a non-finite value")

boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=device)
scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device=device)
_, keep = nms(boxes, scores, 0.5)
torch.cuda.synchronize()
require(keep.numel() == 1, f"MMCV CUDA NMS returned unexpected indices: {keep}")

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}; torchvision: {torchvision.__version__}")
print(f"MMEngine: {mmengine.__version__}; MMCV: {mmcv.__version__}")
print(f"Vendored MMDetection: {mmdet.__version__}")
print(f"FairScale: {version('fairscale')}")
print(f"MMCV compiler: {get_compiler_version()}; CUDA: {get_compiling_cuda_version()}")
print(f"GPU: {name}; capability: {capability}; visible: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print("[OK] PyTorch and MMCV CUDA operations passed on GPU 0")
