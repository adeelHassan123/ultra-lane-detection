import time

import torch


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, input_shape=(1, 3, 256, 256), device="cpu", warmup=10, runs=30) -> float:
    model.eval()
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs
