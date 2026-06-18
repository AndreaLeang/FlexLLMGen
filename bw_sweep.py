import torch, time
import matplotlib.pyplot as plt

sizes_mb = [0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
bws = []
REPS = 50

for mb in sizes_mb:
    n = int(mb * 1024 * 1024 / 4)  # float32 elements
    x = torch.randn(n).pin_memory()
    
    # Warmup
    for _ in range(5):
        y = x.cuda(); torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(REPS):
        y = x.cuda(); torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    bw = x.nbytes * REPS / elapsed / 1e9
    bws.append(bw)
    print(f"{mb:6.1f} MB: {bw:.2f} GB/s")