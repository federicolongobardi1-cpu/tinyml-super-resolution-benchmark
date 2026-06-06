import cv2
import time
import pandas as pd
import numpy as np
from pathlib import Path

from metrics import psnr
from interpolation import downscale, upscale_nearest, upscale_bicubic

INPUT_PATH = Path("data/raw/test2.png")
OUTPUT_DIR = Path("data/outputs")
SCALES= [2, 4]
N_RUNS = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def benchmark_method(method_function, low, target_size, n_runs=50):
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        output = method_function(low, target_size)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / mean_time

    return output, mean_time, std_time, fps

img = cv2.imread(str(INPUT_PATH))

if img is None:
    raise FileNotFoundError(f"Image not found: {INPUT_PATH}")

h, w = img.shape[:2] # Get the height and width of the image
print(f"Original image size: {w}x{h}")
target_size = (w, h)

all_results = []

for scale in SCALES:
    low = downscale(img, scale)

    nearest, nearest_mean, nearest_std, nearest_fps = benchmark_method(
        upscale_nearest,
        low,
        target_size,
        N_RUNS
    )

    bicubic, bicubic_mean, bicubic_std, bicubic_fps = benchmark_method(
        upscale_bicubic,
        low,
        target_size,
        N_RUNS
    )

    cv2.imwrite(str(OUTPUT_DIR / f"low_resolution_x{scale}.png"), low)
    cv2.imwrite(str(OUTPUT_DIR / f"nearest_x{scale}.png"), nearest)
    cv2.imwrite(str(OUTPUT_DIR / f"bicubic_x{scale}.png"), bicubic)

    all_results.append({
        "scale": scale,
        "method": "nearest",
        "psnr": psnr(img, nearest),
        "mean_time_ms": nearest_mean,
        "std_time_ms": nearest_std,
        "fps": nearest_fps
    })

    all_results.append({
        "scale": scale,
        "method": "bicubic",
        "psnr": psnr(img, bicubic),
        "mean_time_ms": bicubic_mean,
        "std_time_ms": bicubic_std,
        "fps": bicubic_fps
    })
df = pd.DataFrame(all_results)
print(df)

df.to_csv("results/benchmark.csv", index=False)