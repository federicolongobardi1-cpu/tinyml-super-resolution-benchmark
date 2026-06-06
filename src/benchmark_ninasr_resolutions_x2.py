"""
Benchmark NiNASR-B0 x2 on different low-resolution input sizes.

This script compares three x2 super-resolution methods:
1. nearest-neighbor interpolation;
2. bicubic interpolation;
3. NiNASR-B0 pretrained model from torchsr.

For each input resolution, the script:
- creates a high-resolution reference image;
- downsamples it to low resolution;
- reconstructs the high-resolution image using each method;
- computes PSNR against the HR reference;
- measures average latency, standard deviation and FPS;
- saves the output images and the final CSV file.
"""

# -----------------------------
# Imports
# -----------------------------

import cv2
import torch
import time
import numpy as np
import pandas as pd
from pathlib import Path

from torchsr.models import ninasr_b0
from metrics import psnr
from interpolation import upscale_nearest, upscale_bicubic


# -----------------------------
# Configuration
# -----------------------------

# Input image used as source for all benchmark resolutions.
INPUT_PATH = Path("data/raw/bird.png")

# Folder where HR, LR and reconstructed images will be saved.
OUTPUT_DIR = Path("data/outputs/ninasr")

# CSV file where benchmark results will be stored.
RESULTS_PATH = Path("results/csv/benchmark_resolutions_ninasr_x2.csv")

# Super-resolution scale factor.
SCALE = 2

# Number of repeated runs used to estimate latency.
# A higher value gives more stable timing results.
N_RUNS = 30

# Low-resolution sizes to test.
# The corresponding HR target size is obtained by multiplying by SCALE.
TEST_RESOLUTIONS = [
    (64, 64),
    (112, 112),
    (224, 224),
    (640, 480),
    (640, 640),
]

# -----------------------------
# Benchmark utility
# -----------------------------

def benchmark_function(function, *args, n_runs=30):
    """
    Measure the execution time of a function.

    Parameters
    ----------
    function : callable
        Function to benchmark.
    *args :
        Arguments passed to the function.
    n_runs : int
        Number of repetitions used for timing.

    Returns
    -------
    output :
        Output of the last function call.
    mean_time : float
        Average execution time in milliseconds.
    std_time : float
        Standard deviation of execution time in milliseconds.
    fps : float
        Frames per second estimated from the average latency.
    """

    times = []

    # Warm-up run.
    # This avoids measuring one-time initialization overhead.
    _ = function(*args)

    for _ in range(n_runs):
        start = time.perf_counter()
        output = function(*args)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / mean_time

    return output, mean_time, std_time, fps


# -----------------------------
# NiNASR inference
# -----------------------------

def ninasr_inference(model, tensor):
    """
    Run NiNASR inference on a single input tensor.

    Input tensor format:
    - shape: [1, C, H, W]
    - range: [0, 1]
    - channel order: RGB

    Output image format:
    - shape: [H, W, C]
    - range: [0, 255]
    - dtype: uint8
    """

    with torch.no_grad():
        output_tensor = model(tensor)

    # Convert from PyTorch tensor format [1, C, H, W]
    # to image format [H, W, C].
    output = output_tensor.squeeze(0)
    output = output.permute(1, 2, 0)

    # Convert tensor to NumPy array.
    output = output.numpy()

    # Clamp model output to valid image range.
    output = np.clip(output, 0, 1)

    # Convert from float [0, 1] to uint8 [0, 255].
    output = (output * 255).astype(np.uint8)

    return output


# -----------------------------
# Image saving utility
# -----------------------------

def save_rgb_image(path, image_rgb):
    """
    Save an RGB image using OpenCV.

    OpenCV saves images in BGR format, so the image is converted
    from RGB to BGR before writing it to disk.
    """

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


# -----------------------------
# Main benchmark
# -----------------------------

def main():
    """
    Execute the full NiNASR x2 benchmark.
    """

    # OpenCV loads images in BGR format.
    img_bgr = cv2.imread(str(INPUT_PATH))

    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {INPUT_PATH}")

    # Convert to RGB because PyTorch image models usually expect RGB input.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Load pretrained NiNASR-B0 model for x2 super-resolution.
    model = ninasr_b0(scale=SCALE, pretrained=True)
    model.eval()

    all_results = []

    for lr_w, lr_h in TEST_RESOLUTIONS:
        # Define the HR reference size corresponding to the LR input.
        hr_w = lr_w * SCALE
        hr_h = lr_h * SCALE

        # Resize the original image to create a controlled HR reference.
        hr = cv2.resize(
            img_rgb,
            (hr_w, hr_h),
            interpolation=cv2.INTER_AREA
        )

        # Generate the LR input from the HR reference.
        lr = cv2.resize(
            hr,
            (lr_w, lr_h),
            interpolation=cv2.INTER_AREA
        )

        target_size = (hr_w, hr_h)

        # Benchmark nearest-neighbor interpolation.
        nearest, nearest_mean, nearest_std, nearest_fps = benchmark_function(
            upscale_nearest,
            lr,
            target_size,
            n_runs=N_RUNS
        )

        # Benchmark bicubic interpolation.
        bicubic, bicubic_mean, bicubic_std, bicubic_fps = benchmark_function(
            upscale_bicubic,
            lr,
            target_size,
            n_runs=N_RUNS
        )

        # Convert LR image to PyTorch tensor.
        # Shape changes from [H, W, C] to [1, C, H, W].
        tensor = torch.from_numpy(lr).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        # Benchmark NiNASR inference.
        ninasr, ninasr_mean, ninasr_std, ninasr_fps = benchmark_function(
            ninasr_inference,
            model,
            tensor,
            n_runs=N_RUNS
        )

        resolution_name = f"{lr_w}x{lr_h}"

        # Save reference, input and reconstructed images for visual inspection.
        save_rgb_image(OUTPUT_DIR / f"hr_{resolution_name}.png", hr)
        save_rgb_image(OUTPUT_DIR / f"lr_{resolution_name}.png", lr)
        save_rgb_image(OUTPUT_DIR / f"nearest_{resolution_name}.png", nearest)
        save_rgb_image(OUTPUT_DIR / f"bicubic_{resolution_name}.png", bicubic)
        save_rgb_image(OUTPUT_DIR / f"ninasr_{resolution_name}.png", ninasr)

        # Store nearest-neighbor result.
        all_results.append({
            "scale": SCALE,
            "lr_resolution": resolution_name,
            "hr_resolution": f"{hr_w}x{hr_h}",
            "method": "nearest_rgb",
            "psnr": psnr(hr, nearest),
            "mean_time_ms": nearest_mean,
            "std_time_ms": nearest_std,
            "fps": nearest_fps,
        })

        # Store bicubic result.
        all_results.append({
            "scale": SCALE,
            "lr_resolution": resolution_name,
            "hr_resolution": f"{hr_w}x{hr_h}",
            "method": "bicubic_rgb",
            "psnr": psnr(hr, bicubic),
            "mean_time_ms": bicubic_mean,
            "std_time_ms": bicubic_std,
            "fps": bicubic_fps,
        })

        # Store NiNASR result.
        all_results.append({
            "scale": SCALE,
            "lr_resolution": resolution_name,
            "hr_resolution": f"{hr_w}x{hr_h}",
            "method": "ninasr_rgb",
            "psnr": psnr(hr, ninasr),
            "mean_time_ms": ninasr_mean,
            "std_time_ms": ninasr_std,
            "fps": ninasr_fps,
        })

    # Convert results to a table and save them as CSV.
    df = pd.DataFrame(all_results)
    print(df)

    df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()