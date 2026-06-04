# Stable Marriage Parallelization — Google Colab Guide

## Quick Start (Copy-Paste into Colab Cells)

### Step 0: Check GPU Availability

> **Before starting**: In Google Colab, go to **Runtime → Change runtime type → T4 GPU** (or any GPU), then click Save.

```python
# Cell 1: Verify GPU is available
!nvidia-smi
```

You should see a Tesla T4 (or similar). If you see an error, change the runtime type.

---

### Step 1: Upload Project Files

```python
# Cell 2: Upload files from your computer
from google.colab import files
uploaded = files.upload()
# Upload: parallel-first.cu, parallel-final.cu, generate_input.py, verify_matching.py
```

Or clone from GitHub if hosted:
```python
# Alternative: Clone from repo
# !git clone <your-repo-url>
```

---

### Step 2: Generate Test Input

```python
# Cell 3: Generate random input with n=10 (try 10, 100, 500, 1000)
!python generate_input.py 10 > input.txt
!echo "=== Generated Input ==="
!cat input.txt
```

---

### Step 3: Compile and Run — Kernel 1 (parallel-first.cu)

```python
# Cell 4: Compile kernel-1 (host-loop version)
!nvcc -o stable_match_v1 parallel-first.cu -arch=sm_75
```

```python
# Cell 5: Run kernel-1
!./stable_match_v1 < input.txt | tee output_v1.txt
```

---

### Step 4: Compile and Run — Kernel 2 (parallel-final.cu)

```python
# Cell 6: Compile kernel-2 (persistent kernel version)
!nvcc -o stable_match_v2 parallel-final.cu -arch=sm_75
```

```python
# Cell 7: Run kernel-2
!./stable_match_v2 < input.txt | tee output_v2.txt
```

---

### Step 5: Verify Correctness

```python
# Cell 8: Verify both outputs produce stable matchings
print("=== Verifying Kernel 1 ===")
!python verify_matching.py input.txt output_v1.txt

print("\n=== Verifying Kernel 2 ===")
!python verify_matching.py input.txt output_v2.txt
```

---

### Step 6: Benchmark Multiple Sizes

```python
# Cell 9: Run benchmarks across different problem sizes
import subprocess

sizes = [10, 50, 100, 200, 500, 1000]
print(f"{'n':>6} | {'Kernel-1 (us)':>15} | {'Kernel-2 (us)':>15}")
print("-" * 45)

for n in sizes:
    # Generate input
    subprocess.run(f"python generate_input.py {n} > input_{n}.txt", shell=True)

    # Run kernel 1
    r1 = subprocess.run(
        f"./stable_match_v1 < input_{n}.txt",
        shell=True, capture_output=True, text=True
    )
    t1 = "ERROR"
    for part in r1.stdout.split(","):
        if "computation" in part:
            t1 = part.split(":")[1].strip().replace(" us", "")

    # Run kernel 2
    r2 = subprocess.run(
        f"./stable_match_v2 < input_{n}.txt",
        shell=True, capture_output=True, text=True
    )
    t2 = "ERROR"
    for part in r2.stdout.split(","):
        if "computation" in part:
            t2 = part.split(":")[1].strip().replace(" us", "")

    print(f"{n:>6} | {t1:>15} | {t2:>15}")
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `nvcc: command not found` | Change runtime to GPU: **Runtime → Change runtime type → T4 GPU** |
| `CUDA error: no kernel image` | Change `-arch=sm_75` to match your GPU. Run `!nvidia-smi` to check. For A100: `-arch=sm_80`, for V100: `-arch=sm_70` |
| `n > 1024` error | The program uses a single thread block (max 1024 threads). Use n ≤ 1024 |
| Program hangs | The persistent kernel (v2) may hang for very large n. Try kernel v1 first |

## Architecture Flags for Common Colab GPUs

| GPU | `-arch` flag |
|-----|-------------|
| Tesla T4 | `sm_75` |
| Tesla V100 | `sm_70` |
| A100 | `sm_80` |
| L4 | `sm_89` |

Run `!nvidia-smi` to identify your GPU.
