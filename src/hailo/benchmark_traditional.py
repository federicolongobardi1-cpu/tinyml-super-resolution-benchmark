import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchmetrics.image import VisualInformationFidelity


INPUT_DIR = Path("data/raw/DIV2K_valid_HR")
OUT_DIR = Path("results")
SUMMARY_CSV = OUT_DIR / "interpolation_summary_vif.csv"

MAX_IMAGES = None
N_RUNS = 30
WARMUP_RUNS = 5

# (LR_H, LR_W, HR_H, HR_W)
RESOLUTIONS = [
    (64, 64, 128, 128),
    (112, 112, 224, 224),
    (224, 224, 448, 448),
    (480, 640, 960, 1280),
    (640, 640, 1280, 1280),
]

METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bicubic": cv2.INTER_CUBIC,
}


def compute_psnr(pred, target):
    mse = np.mean(
        (pred.astype(np.float32) - target.astype(np.float32)) ** 2
    )
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_vif(pred, target, metric):
    pred_t = torch.from_numpy(pred).float() / 255.0
    target_t = torch.from_numpy(target).float() / 255.0

    pred_t = pred_t.unsqueeze(0).unsqueeze(0)
    target_t = target_t.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        value = metric(pred_t, target_t)

    metric.reset()
    return float(value.item())


def center_crop(img, crop_h, crop_w):
    h, w = img.shape

    if h < crop_h or w < crop_w:
        return None

    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2

    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def benchmark_resize(lr, out_h, out_w, interpolation):
    for _ in range(WARMUP_RUNS):
        _ = cv2.resize(lr, (out_w, out_h), interpolation=interpolation)

    times = []

    for _ in range(N_RUNS):
        start = time.perf_counter()
        sr = cv2.resize(lr, (out_w, out_h), interpolation=interpolation)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    latency_ms = times.mean() * 1000.0
    fps = 1.0 / times.mean()

    return sr, latency_ms, fps


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metric = VisualInformationFidelity()

    summary_rows = []
    per_image_rows = []

    image_paths = sorted(INPUT_DIR.glob("*.png"))

    for lr_h, lr_w, hr_h, hr_w in RESOLUTIONS:
        for method_name, interpolation in METHODS.items():

            psnr_values = []
            vif_values = []
            fps_values = []
            latency_values = []

            valid_images = 0

            for path in image_paths:
                if MAX_IMAGES is not None and valid_images >= MAX_IMAGES:
                    break

                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                hr = center_crop(img, hr_h, hr_w)

                if hr is None:
                    print(
                        f"Skipping {path.name}: "
                        f"too small for {hr_w}x{hr_h}"
                    )
                    continue

                lr = cv2.resize(
                    hr,
                    (lr_w, lr_h),
                    interpolation=cv2.INTER_CUBIC,
                )

                sr, latency_ms, fps = benchmark_resize(
                    lr,
                    hr_h,
                    hr_w,
                    interpolation,
                )

                psnr = compute_psnr(sr, hr)
                vif = compute_vif(sr, hr, metric)

                psnr_values.append(psnr)
                vif_values.append(vif)
                fps_values.append(fps)
                latency_values.append(latency_ms)

                per_image_rows.append({
                    "image": path.name,
                    "method": method_name,
                    "lr_resolution": f"{lr_w}x{lr_h}",
                    "hr_resolution": f"{hr_w}x{hr_h}",
                    "psnr": psnr,
                    "vif": vif,
                    "latency_ms": latency_ms,
                    "fps": fps,
                })

                valid_images += 1

            if valid_images == 0:
                raise RuntimeError(
                    f"No valid images for resolution "
                    f"{lr_w}x{lr_h} -> {hr_w}x{hr_h}"
                )

            psnr_values = np.array(psnr_values)
            vif_values = np.array(vif_values)
            fps_values = np.array(fps_values)
            latency_values = np.array(latency_values)

            summary_rows.append({
                "method": method_name,
                "lr_resolution": f"{lr_w}x{lr_h}",
                "hr_resolution": f"{hr_w}x{hr_h}",
                "n_images": valid_images,
                "psnr_mean": psnr_values.mean(),
                "psnr_std": psnr_values.std(),
                "vif_mean": vif_values.mean(),
                "vif_std": vif_values.std(),
                "latency_ms_mean": latency_values.mean(),
                "latency_ms_std": latency_values.std(),
                "fps_mean": fps_values.mean(),
                "fps_std": fps_values.std(),
            })

            print(
                f"{method_name:8s} | "
                f"{lr_w}x{lr_h} -> {hr_w}x{hr_h} | "
                f"PSNR {psnr_values.mean():.3f} ± "
                f"{psnr_values.std():.3f} dB | "
                f"VIF {vif_values.mean():.4f} ± "
                f"{vif_values.std():.4f} | "
                f"FPS {fps_values.mean():.2f} ± "
                f"{fps_values.std():.2f}"
            )

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "lr_resolution",
                "hr_resolution",
                "n_images",
                "psnr_mean",
                "psnr_std",
                "vif_mean",
                "vif_std",
                "latency_ms_mean",
                "latency_ms_std",
                "fps_mean",
                "fps_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    per_image_csv = OUT_DIR / "interpolation_per_image_vif.csv"

    with open(per_image_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "method",
                "lr_resolution",
                "hr_resolution",
                "psnr",
                "vif",
                "latency_ms",
                "fps",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(f"Saved summary results to:   {SUMMARY_CSV}")
    print(f"Saved per-image results to: {per_image_csv}")


if __name__ == "__main__":
    main()