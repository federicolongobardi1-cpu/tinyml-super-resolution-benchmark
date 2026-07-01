import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchmetrics.image import VisualInformationFidelity

from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


HEF_PATH = Path("weights/espcn_x2_finetuned_640_fixed.hef")
INPUT_DIR = Path("data/raw/DIV2K_valid_HR")

N_RUNS = 30
WARMUP_RUNS = 5


def psnr(pred, target):
    mse = np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2)
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


def get_hw_from_shape(shape):
    dims = list(shape)

    if len(dims) == 4:
        return int(dims[1]), int(dims[2])
    if len(dims) == 3:
        return int(dims[0]), int(dims[1])

    raise RuntimeError(f"Unsupported vstream shape: {shape}")


def preprocess_lr(lr):
    x = lr.astype(np.float32) / 255.0
    x = x[np.newaxis, :, :, np.newaxis]  # NHWC
    return x


def postprocess(output):
    sr = np.squeeze(output)
    sr = np.clip(sr, 0.0, 1.0)
    sr = (sr * 255.0).round().astype(np.uint8)
    return sr


def main():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    metric = VisualInformationFidelity()

    hef = HEF(str(HEF_PATH))

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,
        )

        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        output_vstreams_params = OutputVStreamParams.make(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        input_info = hef.get_input_vstream_infos()[0]
        output_info = hef.get_output_vstream_infos()[0]

        input_name = input_info.name
        output_name = output_info.name

        lr_h, lr_w = get_hw_from_shape(input_info.shape)
        hr_h, hr_w = get_hw_from_shape(output_info.shape)

        print("Input:", input_name, input_info.shape)
        print("Output:", output_name, output_info.shape)
        print(f"Benchmark resolution: {lr_w}x{lr_h} -> {hr_w}x{hr_h}")

        psnr_values = []
        vif_values = []
        latency_values = []
        fps_values = []
        valid_images = 0

        image_paths = sorted(INPUT_DIR.glob("*.png"))

        with InferVStreams(
            network_group,
            input_vstreams_params,
            output_vstreams_params,
        ) as infer_pipeline:
            with network_group.activate(network_group_params):

                warmup_sample = None

                for path in image_paths:
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    hr = center_crop(img, hr_h, hr_w)
                    if hr is None:
                        continue

                    lr = cv2.resize(
                        hr,
                        (lr_w, lr_h),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    warmup_sample = preprocess_lr(lr)
                    break

                if warmup_sample is None:
                    raise RuntimeError(
                        f"No valid images large enough for {hr_w}x{hr_h}"
                    )

                for _ in range(WARMUP_RUNS):
                    _ = infer_pipeline.infer({input_name: warmup_sample})

                for path in image_paths:
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    hr = center_crop(img, hr_h, hr_w)
                    if hr is None:
                        print(
                            f"Skipping {path.name}: too small for {hr_w}x{hr_h}",
                            flush=True,
                        )
                        continue

                    lr = cv2.resize(
                        hr,
                        (lr_w, lr_h),
                        interpolation=cv2.INTER_CUBIC,
                    )

                    input_data = preprocess_lr(lr)

                    times = []
                    result = None

                    for _ in range(N_RUNS):
                        start = time.perf_counter()
                        result = infer_pipeline.infer({input_name: input_data})
                        end = time.perf_counter()
                        times.append(end - start)

                    output = result[output_name]
                    sr = postprocess(output)

                    if sr.shape != hr.shape:
                        raise RuntimeError(
                            f"Shape mismatch for {path.name}: SR {sr.shape}, HR {hr.shape}"
                        )

                    score = psnr(sr, hr)
                    vif = compute_vif(sr, hr, metric)

                    times = np.array(times)
                    latency_ms = times.mean() * 1000.0
                    fps = 1.0 / times.mean()

                    psnr_values.append(score)
                    vif_values.append(vif)
                    latency_values.append(latency_ms)
                    fps_values.append(fps)

                    valid_images += 1

                    print(
                        f"{lr_w}x{lr_h} -> {hr_w}x{hr_h} | "
                        f"{path.name} | image {valid_images} | "
                        f"PSNR {score:.3f} dB | VIF {vif:.4f} | FPS {fps:.2f}",
                        flush=True,
                    )

        psnr_values = np.array(psnr_values)
        vif_values = np.array(vif_values)
        latency_values = np.array(latency_values)
        fps_values = np.array(fps_values)

        row = {
            "method": "espcn_hailo",
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
        }

        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        print("\nSUMMARY")
        print("-------")
        print("method:          espcn_hailo")
        print(f"resolution:      {lr_w}x{lr_h} -> {hr_w}x{hr_h}")
        print(f"images:          {valid_images}")
        print(f"PSNR:            {psnr_values.mean():.3f} ± {psnr_values.std():.3f} dB")
        print(f"VIF:             {vif_values.mean():.4f} ± {vif_values.std():.4f}")
        print(f"latency:         {latency_values.mean():.3f} ± {latency_values.std():.3f} ms")
        print(f"FPS:             {fps_values.mean():.3f} ± {fps_values.std():.3f}")
        print(f"CSV saved:       {CSV_PATH}")


if __name__ == "__main__":
    main()