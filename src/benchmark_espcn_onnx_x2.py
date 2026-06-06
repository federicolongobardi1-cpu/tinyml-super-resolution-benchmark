"""
Benchmark ESPCN x2 using ONNX Runtime.

This script compares:
- ESPCN PyTorch
- ESPCN ONNX Runtime

The goal is to verify that the exported ONNX model gives comparable PSNR
and to measure whether ONNX Runtime improves inference speed.
"""

import cv2
import time
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

from models.espcn import ESPCN
from metrics import psnr


INPUT_PATH = Path("data/raw/bird.png")
WEIGHTS_PATH = Path("weights/ESPCN_x2.pth.tar")
ONNX_PATH = Path("weights/espcn_x2_dynamic.onnx")

OUTPUT_DIR = Path("data/outputs/espcn/onnx_x2")
RESULTS_PATH = Path("results/csv/benchmark_espcn_onnx_x2.csv")

SCALE = 2
N_RUNS = 30

TEST_RESOLUTIONS = [
    (64, 64),
    (112, 112),
    (224, 224),
    (640, 480),
    (640, 640),
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def benchmark_function(function, *args, n_runs=30):
    times = []

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


def load_pytorch_model():
    model = ESPCN(upscale_factor=SCALE, in_channels=1, out_channels=1)

    checkpoint = torch.load(str(WEIGHTS_PATH), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def pytorch_inference(model, tensor):
    with torch.no_grad():
        output_tensor = model(tensor)

    output = output_tensor.squeeze().numpy()
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    return output


def onnx_inference(session, tensor_np):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor_np})[0]

    output = np.squeeze(output)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    return output


def main():
    img = cv2.imread(str(INPUT_PATH))

    if img is None:
        raise FileNotFoundError(f"Image not found: {INPUT_PATH}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pytorch_model = load_pytorch_model()

    onnx_session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    all_results = []

    for lr_w, lr_h in TEST_RESOLUTIONS:
        hr_w = lr_w * SCALE
        hr_h = lr_h * SCALE

        hr = cv2.resize(gray, (hr_w, hr_h), interpolation=cv2.INTER_AREA)
        lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(lr).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        tensor_np = tensor.numpy().astype(np.float32)

        pytorch_out, pytorch_mean, pytorch_std, pytorch_fps = benchmark_function(
            pytorch_inference,
            pytorch_model,
            tensor,
            n_runs=N_RUNS
        )

        onnx_out, onnx_mean, onnx_std, onnx_fps = benchmark_function(
            onnx_inference,
            onnx_session,
            tensor_np,
            n_runs=N_RUNS
        )

        resolution_name = f"{lr_w}x{lr_h}"

        cv2.imwrite(str(OUTPUT_DIR / f"hr_{resolution_name}.png"), hr)
        cv2.imwrite(str(OUTPUT_DIR / f"lr_{resolution_name}.png"), lr)
        cv2.imwrite(str(OUTPUT_DIR / f"pytorch_{resolution_name}.png"), pytorch_out)
        cv2.imwrite(str(OUTPUT_DIR / f"onnx_{resolution_name}.png"), onnx_out)

        all_results.append({
            "scale": SCALE,
            "lr_resolution": resolution_name,
            "hr_resolution": f"{hr_w}x{hr_h}",
            "method": "espcn_pytorch",
            "psnr": psnr(hr, pytorch_out),
            "mean_time_ms": pytorch_mean,
            "std_time_ms": pytorch_std,
            "fps": pytorch_fps,
        })

        all_results.append({
            "scale": SCALE,
            "lr_resolution": resolution_name,
            "hr_resolution": f"{hr_w}x{hr_h}",
            "method": "espcn_onnx",
            "psnr": psnr(hr, onnx_out),
            "mean_time_ms": onnx_mean,
            "std_time_ms": onnx_std,
            "fps": onnx_fps,
        })

        print(f"\nResolution {resolution_name}")
        print(f"PyTorch PSNR: {psnr(hr, pytorch_out):.4f} dB")
        print(f"ONNX PSNR:    {psnr(hr, onnx_out):.4f} dB")
        print(f"PyTorch FPS:  {pytorch_fps:.2f}")
        print(f"ONNX FPS:     {onnx_fps:.2f}")

    df = pd.DataFrame(all_results)
    print(df)

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()