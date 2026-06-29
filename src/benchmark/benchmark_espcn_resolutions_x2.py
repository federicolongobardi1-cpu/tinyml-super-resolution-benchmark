"""
Benchmark ESPCN x2 on different low-resolution input sizes.

This script compares three x2 super-resolution methods:
1. nearest-neighbor interpolation;
2. bicubic interpolation;
3. ESPCN pretrained model.

For each input resolution, the script:
- creates a high-resolution grayscale reference image;
- downsamples it to low resolution;
- reconstructs the high-resolution image using each method;
- computes PSNR against the HR reference;
- measures average latency, standard deviation and FPS;
- saves the output images and the final CSV file.

Note:
This ESPCN implementation works on grayscale images, while the NiNASR
benchmark currently works on RGB images. Therefore, their PSNR values are
not directly comparable unless both are evaluated with the same color format.
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

from models.espcn import ESPCN
from metrics import psnr
from interpolation import upscale_nearest, upscale_bicubic


# -----------------------------
# Configuration
# -----------------------------

INPUT_PATH = Path("data/raw/bird.png")
WEIGHTS_PATH = Path("weights/ESPCN_x2.pth.tar")

OUTPUT_DIR = Path("data/outputs/espcn")
RESULTS_PATH = Path("results/csv/benchmark_resolutions_espcn_x2.csv")

SCALE = 2
N_RUNS = 30

# Low-resolution sizes to test.
# The corresponding HR size is obtained by multiplying by SCALE.
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
# ESPCN inference
# -----------------------------

def espcn_inference(model, tensor):
    """
    Run ESPCN inference on a single grayscale input tensor.

    Input tensor format:
    - shape: [1, 1, H, W]
    - range: [0, 1]

    Output image format:
    - shape: [H, W]
    - range: [0, 255]
    - dtype: uint8
    """

    with torch.no_grad():
        output_tensor = model(tensor)

    # Remove batch and channel dimensions.
    output = output_tensor.squeeze().numpy()

    # Clamp model output to valid image range.
    output = np.clip(output, 0, 1)

    # Convert from float [0, 1] to uint8 [0, 255].
    output = (output * 255).astype(np.uint8)

    return output


# -----------------------------
# Main benchmark
# -----------------------------

def main():
    """
    Execute the full ESPCN x2 benchmark.
    """

    # OpenCV loads images in BGR format.
    img = cv2.imread(str(INPUT_PATH))

    if img is None:
        raise FileNotFoundError(f"Image not found: {INPUT_PATH}")

    # Convert to grayscale because this ESPCN model uses one input channel.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create ESPCN x2 model and load pretrained weights.
    model = ESPCN(upscale_factor=SCALE, in_channels=1, out_channels=1)

    checkpoint = torch.load(str(WEIGHTS_PATH), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_results = []

    for lr_w, lr_h in TEST_RESOLUTIONS:
        # Define HR reference size corresponding to the LR input.
        hr_w = lr_w * SCALE
        hr_h = lr_h * SCALE

        # Resize the original image to create a controlled HR reference.
        hr = cv2.resize(
            gray,
            (hr_w, hr_h),
            interpolation=cv2.INTER_AREA
        )

        # Generate LR input from the HR reference.
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
        # Shape changes from [H, W] to [1, 1, H, W].
        tensor = torch.from_numpy(lr).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # Benchmark ESPCN inference.
        espcn, espcn_mean, espcn_std, espcn_fps = benchmark_function(
            espcn_inference,
            model,
            tensor,
            n_runs=N_RUNS
        )

        resolution_name = f"{lr_w}x{lr_h}"

        # Save reference, input and reconstructed images.
        cv2.imwrite(str(OUTPUT_DIR / f"hr_{resolution_name}.png"), hr)
        cv2.imwrite(str(OUTPUT_DIR / f"lr_{resolution_name}.png"), lr)
        cv2.imwrite(str(OUTPUT_DIR / f"nearest_{resolution_name}.png"), nearest)
        cv2.imwrite(str(OUTPUT_DIR / f"bicubic_{resolution_name}.png"), bicubic)
        cv2.imwrite(str(OUTPUT_DIR / f"espcn_{resolution_name}.png"), espcn)

        # Store nearest-neighbor result.
        all_results.append({
            "scale": SCALE,
            "lr_resolution": resolution_name,
            "hr_resolution": f"{hr_w}x{hr_h}",
            "method": "nearest_gray",
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
            "method": "bicubic_gray",
            "psnr": psnr(hr, bicubic),
            "mean_time_ms": bicubic_mean,
            "std_time_ms": bicubic_std,
            "fps": bicubic_fps,
        })

        # Store ESPCN result.
        all_results.append({
            "scale": SCALE,
            "lr_resolution": resolution_name,
            "hr_resolution": f"{hr_w}x{hr_h}",
            "method": "espcn_gray",
            "psnr": psnr(hr, espcn),
            "mean_time_ms": espcn_mean,
            "std_time_ms": espcn_std,
            "fps": espcn_fps,
        })

    # Convert results to a table and save them as CSV.
    df = pd.DataFrame(all_results)
    print(df)

    df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()