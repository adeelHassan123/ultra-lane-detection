"""
Benchmark script to measure training speed and GPU utilization.
Run before and after optimizations to verify improvements.
"""

import time
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from configs.cnn_config import CNNConfig
from data.dataset import TuSimpleLaneDataset
from data.fast_dataset import FastLaneDataset
from data.transforms import build_transforms
from models import get_model
from training.train_epoch import train_one_epoch


def benchmark_dataloader(data_root, batch_size=16, num_batches=10):
    """Benchmark data loading speed."""
    print("\n" + "="*60)
    print("DATALOADER BENCHMARK")
    print("="*60)
    
    # Test regular dataset
    try:
        regular_ds = TuSimpleLaneDataset(
            str(Path(data_root) / "train" / "images"),
            str(Path(data_root) / "train" / "masks"),
            transform=build_transforms(256, "none")
        )
        regular_loader = DataLoader(regular_ds, batch_size=batch_size, num_workers=2)
        
        start = time.time()
        for i, (img, mask) in enumerate(regular_loader):
            if i >= num_batches:
                break
        regular_time = time.time() - start
        regular_speed = (num_batches * batch_size) / regular_time
        
        print(f"Regular Dataset (JPEG/PNG):")
        print(f"   Time for {num_batches} batches: {regular_time:.2f}s")
        print(f"   Speed: {regular_speed:.1f} samples/sec")
    except Exception as e:
        print(f"Regular dataset failed: {e}")
        regular_time, regular_speed = 0, 0
    
    # Test fast dataset (if exists)
    fast_root = Path(data_root).parent / "fast_data"
    if (fast_root / "train" / "images").exists():
        try:
            fast_ds = FastLaneDataset(str(fast_root), split='train', transform=build_transforms(256, "none"))
            fast_loader = DataLoader(fast_ds, batch_size=batch_size, num_workers=2)
            
            start = time.time()
            for i, (img, mask) in enumerate(fast_loader):
                if i >= num_batches:
                    break
            fast_time = time.time() - start
            fast_speed = (num_batches * batch_size) / fast_time
            
            print(f"\nFast Dataset (NumPy):")
            print(f"   Time for {num_batches} batches: {fast_time:.2f}s")
            print(f"   Speed: {fast_speed:.1f} samples/sec")
            
            if regular_time > 0:
                speedup = regular_time / fast_time
                print(f"\nSPEEDUP: {speedup:.1f}x faster with numpy format")
        except Exception as e:
            print(f"Fast dataset not available: {e}")
    else:
        print(f"\n   (Run scripts/preprocess_to_numpy.py to test fast dataset)")


def benchmark_gpu_utilization(batch_size=16, num_batches=20):
    """Benchmark GPU utilization during training."""
    print("\n" + "="*60)
    print("GPU UTILIZATION BENCHMARK")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("[ERROR] No GPU available")
        return
    
    # Create dummy data
    dummy_images = torch.randn(batch_size, 3, 256, 256, device=device)
    dummy_masks = torch.randn(batch_size, 1, 256, 256, device=device)
    
    # Create model
    cfg = CNNConfig()
    model = get_model(cfg).to(device)
    
    # Warmup
    for _ in range(5):
        _ = model(dummy_images)
    
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(num_batches):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        logits = model(dummy_images)
        loss = (logits - dummy_masks).mean()
        loss.backward()
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds
    
    avg_time = sum(times) / len(times)
    throughput = (batch_size * 1000) / avg_time  # samples/sec
    
    print(f"Model forward+backward:")
    print(f"   Batch size: {batch_size}")
    print(f"   Avg time per batch: {avg_time:.2f} ms")
    print(f"   Throughput: {throughput:.1f} samples/sec")
    print(f"   GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/content/local_data")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    print("="*60)
    print("ULTRA LANE DETECTION - PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Check paths
    if not Path(args.data_root).exists():
        print(f"\n[ERROR] Data root not found: {args.data_root}")
        print("   Run first: python scripts/setup_colab.py")
        return 1
    
    benchmark_dataloader(args.data_root, args.batch_size)
    benchmark_gpu_utilization(args.batch_size)
    
    print("\n" + "="*60)
    print("[OK] BENCHMARK COMPLETE")
    print("="*60)
    print("\nExpected performance with optimizations:")
    print("   - DataLoader: >100 samples/sec with numpy format")
    print("   - GPU: >50 samples/sec throughput")
    print("   - Epoch time: 2-3 minutes")
    print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
