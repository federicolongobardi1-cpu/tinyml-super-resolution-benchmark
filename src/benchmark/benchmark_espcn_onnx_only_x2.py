"""
Benchmark ESPCN x2 using only ONNX Runtime.

This script does not import PyTorch. It is intended for final
latency, FPS and power measurements on Raspberry Pi 5.
"""

import cv2
import time
import argparse
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

from metrics import psnr


INPUT_PATH = Path("data/raw/bird.png")
ONNX_PATH = Path("weights/espcn_x2_dynamic_int8.onnx")

OUTPUT_DIR = Path("data/outputs/espcn/onnx_only_x2")
RESULTS_PATH = Path("results/csv/benchmark_espcn_onnx_only_x2.csv")

SCALE = 2
N_RUNS = 50

TEST_RESOLUTIONS = [
    (64, 64),
    (112, 112),
    (224, 224),
    (640, 480),
    (640, 640),
]


def benchmark_function(function, *args, n_runs=50):
    times = []

    # Warm-up
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


def onnx_inference(session, input_name, tensor_np):
    output = session.run(None, {input_name: tensor_np})[0]

    output = np.squeeze(output)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    return output


def prepare_tensor(lr):
    tensor = lr.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, np.newaxis, :, :]
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=str,
        default="all",
        help="LR resolution to test, e.g. 224x224, 640x640, or all"
    )

    parser.add_argument(
    "--runs",
    type=int,
    default=N_RUNS,
    help="Number of inference runs"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(INPUT_PATH))

    if img is None:
        raise FileNotFoundError(f"Image not found: {INPUT_PATH}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    all_results = []

    for lr_w, lr_h in TEST_RESOLUTIONS:
        resolution_name = f"{lr_w}x{lr_h}"

        if args.resolution != "all" and args.resolution != resolution_name:
            continue

        hr_w = lr_w * SCALE
        hr_h = lr_h * SCALE

        hr = cv2.resize(gray, (hr_w, hr_h), interpolation=cv2.INTER_AREA)
        lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

        tensor_np = prepare_tensor(lr)

        onnx_out, mean_time, std_time, fps = benchmark_function(
            onnx_inference,
            session,
            input_name,
            tensor_np,
            n_runs=args.runs
        )

        cv2.imwrite(str(OUTPUT_DIR / f"hr_{resolution_name}.png"), hr)
        cv2.imwrite(str(OUTPUT_DIR / f"lr_{resolution_name}.png"), lr)
        cv2.imwrite(str(OUTPUT_DIR / f"onnx_{resolution_name}.png"), onnx_out)

        result = {
            "scale": SCALE,
            "lr_resolution": resolution_name,
            "hr_resolution": f"{hr_w}x{hr_h}",
            "method": "espcn_onnx_only",
            "psnr": psnr(hr, onnx_out),
            "mean_time_ms": mean_time,
            "std_time_ms": std_time,
            "fps": fps,
        }

        all_results.append(result)

        print(f"\nResolution {resolution_name}")
        print(f"PSNR: {result['psnr']:.4f} dB")
        print(f"Mean time: {mean_time:.4f} ms")
        print(f"Std time: {std_time:.4f} ms")
        print(f"FPS: {fps:.2f}")

    df = pd.DataFrame(all_results)
    print(df)

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()