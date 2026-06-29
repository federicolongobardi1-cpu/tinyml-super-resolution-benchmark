import time
import csv
from pathlib import Path

import cv2
import numpy as np

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


# =========================
# CONFIG
# =========================

MODEL_SIZE = 640  # use 224 or 640

INPUT_DIR = Path("data/raw/DIV2K_valid_HR")
N_IMAGES = 30
WARMUP_RUNS = 5

# =========================


if MODEL_SIZE == 224:
    HEF_PATH = Path("weights/espcn_x2_finetuned_224_fixed.hef")
    OUTPUT_DIR = Path("data/outputs/hailo_eval_sr_224")
    CSV_PATH = Path("results/hailo_eval_sr_224.csv")
    LR_SIZE = 224
    HR_SIZE = 448

elif MODEL_SIZE == 640:
    HEF_PATH = Path("weights/espcn_x2_finetuned_640_fixed.hef")
    OUTPUT_DIR = Path("data/outputs/hailo_eval_sr_640")
    CSV_PATH = Path("results/hailo_eval_sr_640.csv")
    LR_SIZE = 640
    HR_SIZE = 1280

else:
    raise ValueError("MODEL_SIZE must be 224 or 640")


def preprocess_from_hr(hr_crop):
    lr = cv2.resize(
        hr_crop,
        (LR_SIZE, LR_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )

    x = lr.astype(np.float32) / 255.0
    x = x[np.newaxis, :, :, np.newaxis]  # NHWC for Hailo

    return x, lr


def postprocess(output):
    y = np.squeeze(output)
    y = np.clip(y, 0.0, 1.0)
    y = (y * 255.0).astype(np.uint8)
    return y


def psnr(pred, target):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    mse = np.mean((pred - target) ** 2)

    if mse == 0:
        return float("inf")

    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def load_samples():
    samples = []

    for path in sorted(INPUT_DIR.glob("*.png")):
        if len(samples) >= N_IMAGES:
            break

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        h, w = img.shape

        if h < HR_SIZE or w < HR_SIZE:
            print(f"Skipping {path.name}: too small ({w}x{h})")
            continue

        y0 = (h - HR_SIZE) // 2
        x0 = (w - HR_SIZE) // 2

        hr = img[y0:y0 + HR_SIZE, x0:x0 + HR_SIZE]

        x, lr = preprocess_from_hr(hr)

        samples.append((path, hr, lr, x))

    if len(samples) == 0:
        raise RuntimeError(
            f"No valid images found in {INPUT_DIR} for HR_SIZE={HR_SIZE}"
        )

    return samples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Model size:", MODEL_SIZE)
    print("HEF:", HEF_PATH)
    print("LR size:", LR_SIZE)
    print("HR size:", HR_SIZE)

    samples = load_samples()

    hef = HEF(str(HEF_PATH))

    rows = []
    latencies = []
    psnrs = []

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,
        )

        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_params = InputVStreamParams.make(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        output_params = OutputVStreamParams.make(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        input_info = hef.get_input_vstream_infos()[0]
        output_info = hef.get_output_vstream_infos()[0]

        input_name = input_info.name
        output_name = output_info.name

        print("Input:", input_name, input_info.shape)
        print("Output:", output_name, output_info.shape)
        print("Images:", len(samples))

        with InferVStreams(
            network_group,
            input_params,
            output_params,
        ) as infer_pipeline:
            with network_group.activate(network_group_params):
                for i in range(WARMUP_RUNS):
                    _, _, _, x = samples[i % len(samples)]
                    _ = infer_pipeline.infer({input_name: x})

                for path, hr, lr, x in samples:
                    start = time.perf_counter()
                    results = infer_pipeline.infer({input_name: x})
                    end = time.perf_counter()

                    sr = postprocess(results[output_name])

                    if sr.shape != hr.shape:
                        raise RuntimeError(
                            f"Shape mismatch for {path.name}: "
                            f"sr={sr.shape}, hr={hr.shape}"
                        )

                    latency_ms = (end - start) * 1000.0
                    score = psnr(sr, hr)

                    latencies.append(latency_ms)
                    psnrs.append(score)

                    cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}_hr.png"), hr)
                    cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}_lr.png"), lr)
                    cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}_hailo_sr.png"), sr)

                    rows.append({
                        "image": path.name,
                        "latency_ms": latency_ms,
                        "psnr": score,
                    })

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "latency_ms", "psnr"],
        )
        writer.writeheader()
        writer.writerows(rows)

    latencies = np.array(latencies)
    psnrs = np.array(psnrs)

    print("\nEvaluation results")
    print("------------------")
    print(f"Images:        {len(samples)}")
    print(f"Mean latency:  {latencies.mean():.3f} ms")
    print(f"Std latency:   {latencies.std():.3f} ms")
    print(f"FPS:           {1000.0 / latencies.mean():.2f}")
    print(f"Mean PSNR:     {psnrs.mean():.3f} dB")
    print(f"Std PSNR:      {psnrs.std():.3f} dB")
    print(f"CSV saved:     {CSV_PATH}")
    print(f"Images saved:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()