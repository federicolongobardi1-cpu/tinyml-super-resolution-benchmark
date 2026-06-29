import csv
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


INPUT_DIR = Path("data/raw/DIV2K_valid_HR")
ONNX_PATH = Path("weights/Finetuned/640x640/espcn_x2_finetuned_640_dynamic.onnx")
CSV_PATH = Path("results/espcn_onnx_summary.csv")

N_RUNS = 30
WARMUP_RUNS = 5

RESOLUTIONS = [
    (64, 64, 128, 128),
    (112, 112, 224, 224),
    (224, 224, 448, 448),
    (480, 640, 960, 1280),
    (640, 640, 1280, 1280),
]


def psnr(pred, target):
    mse = np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def center_crop(img, crop_h, crop_w):
    h, w = img.shape

    if h < crop_h or w < crop_w:
        return None

    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2

    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def benchmark(session, input_name, lr):
    x = lr.astype(np.float32) / 255.0
    x = x[np.newaxis, np.newaxis, :, :]

    for _ in range(WARMUP_RUNS):
        _ = session.run(None, {input_name: x})

    times = []

    for _ in range(N_RUNS):
        start = time.perf_counter()
        y = session.run(None, {input_name: x})[0]
        end = time.perf_counter()
        times.append(end - start)

    sr = np.squeeze(y)
    sr = np.clip(sr, 0.0, 1.0)
    sr = (sr * 255.0).round().astype(np.uint8)

    times = np.array(times)

    latency_ms = times.mean() * 1000.0
    fps = 1.0 / times.mean()

    return sr, latency_ms, fps


def main():
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(INPUT_DIR.glob("*.png"))
    summary_rows = []

    for lr_h, lr_w, hr_h, hr_w in RESOLUTIONS:
        psnr_values = []
        latency_values = []
        fps_values = []

        valid_images = 0

        for path in image_paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            hr = center_crop(img, hr_h, hr_w)

            if hr is None:
                print(f"Skipping {path.name}: too small for {hr_w}x{hr_h}", flush=True)
                continue

            lr = cv2.resize(
                hr,
                (lr_w, lr_h),
                interpolation=cv2.INTER_CUBIC
            )

            sr, latency_ms, fps = benchmark(session, input_name, lr)

            if sr.shape != hr.shape:
                raise RuntimeError(
                    f"Shape mismatch for {path.name}: SR {sr.shape}, HR {hr.shape}"
                )

            score = psnr(sr, hr)

            psnr_values.append(score)
            latency_values.append(latency_ms)
            fps_values.append(fps)

            valid_images += 1

            print(
                f"{lr_w}x{lr_h} -> {hr_w}x{hr_h} | "
                f"{path.name} | image {valid_images} | "
                f"PSNR {score:.3f} dB | FPS {fps:.2f}",
                flush=True
            )

        if valid_images == 0:
            raise RuntimeError(
                f"No valid images for resolution {lr_w}x{lr_h} -> {hr_w}x{hr_h}"
            )

        psnr_values = np.array(psnr_values)
        latency_values = np.array(latency_values)
        fps_values = np.array(fps_values)

        summary_rows.append({
            "method": "espcn_onnx",
            "lr_resolution": f"{lr_w}x{lr_h}",
            "hr_resolution": f"{hr_w}x{hr_h}",
            "n_images": valid_images,
            "psnr_mean": psnr_values.mean(),
            "psnr_std": psnr_values.std(),
            "latency_ms_mean": latency_values.mean(),
            "latency_ms_std": latency_values.std(),
            "fps_mean": fps_values.mean(),
            "fps_std": fps_values.std(),
        })

        print(
            f"\nSUMMARY | espcn_onnx | {lr_w}x{lr_h} -> {hr_w}x{hr_h} | "
            f"n={valid_images} | "
            f"PSNR {psnr_values.mean():.3f} ± {psnr_values.std():.3f} dB | "
            f"Latency {latency_values.mean():.3f} ± {latency_values.std():.3f} ms | "
            f"FPS {fps_values.mean():.3f} ± {fps_values.std():.3f}\n",
            flush=True
        )

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "lr_resolution",
                "hr_resolution",
                "n_images",
                "psnr_mean",
                "psnr_std",
                "latency_ms_mean",
                "latency_ms_std",
                "fps_mean",
                "fps_std",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary saved to {CSV_PATH}")


if __name__ == "__main__":
    main()