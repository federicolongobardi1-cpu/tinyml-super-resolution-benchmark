import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort

from hailo_platform import (
    HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from models.espcn import ESPCN


INPUT_IMAGE = Path("data/raw/DIV2K_valid_HR/0889.png")

PYTORCH_WEIGHTS = Path("weights/espcn_x2_finetuned_224_stage2.pth.tar")
ONNX_PATH = Path("weights/espcn_x2_finetuned_224_dynamic.onnx")
HEF_PATH = Path("weights/espcn_x2_finetuned_224_fixed.hef")

OUT_PATH = Path("results/plots/quality_comparison_224.png")

LR_SIZE = 224
HR_SIZE = 448
SCALE = 2


def psnr(pred, target):
    mse = np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def center_crop(img, size):
    h, w = img.shape
    if h < size or w < size:
        raise RuntimeError(f"Image too small: {w}x{h}, required {size}x{size}")
    y0 = max(0, (h - size) // 2 - 550)
    x0 = max(0, (w - size) // 2 -100)
    return img[y0:y0 + size, x0:x0 + size]


def postprocess(y):
    sr = np.squeeze(y)
    sr = np.clip(sr, 0.0, 1.0)
    sr = (sr * 255.0).round().astype(np.uint8)
    return sr


def load_pytorch_model():
    model = ESPCN(upscale_factor=SCALE, in_channels=1, out_channels=1)
    checkpoint = torch.load(PYTORCH_WEIGHTS, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def run_pytorch(model, lr):
    x = lr.astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        y = model(x)

    return postprocess(y.squeeze().cpu().numpy())


def run_onnx(lr):
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    x = lr.astype(np.float32) / 255.0
    x = x[np.newaxis, np.newaxis, :, :]

    y = session.run(None, {input_name: x})[0]
    return postprocess(y)


def run_hailo(lr):
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

        input_name = hef.get_input_vstream_infos()[0].name
        output_name = hef.get_output_vstream_infos()[0].name

        x = lr.astype(np.float32) / 255.0
        x = x[np.newaxis, :, :, np.newaxis]  # NHWC

        with InferVStreams(
            network_group,
            input_vstreams_params,
            output_vstreams_params,
        ) as infer_pipeline:
            with network_group.activate(network_group_params):
                result = infer_pipeline.infer({input_name: x})

        return postprocess(result[output_name])


def add_label(img, text):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    canvas = img.copy()

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {INPUT_IMAGE}")

    hr = center_crop(img, HR_SIZE)

    lr = cv2.resize(
        hr,
        (LR_SIZE, LR_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )

    lr_vis = cv2.resize(lr, (HR_SIZE, HR_SIZE), interpolation=cv2.INTER_NEAREST)
    nearest = cv2.resize(lr, (HR_SIZE, HR_SIZE), interpolation=cv2.INTER_NEAREST)
    bicubic = cv2.resize(lr, (HR_SIZE, HR_SIZE), interpolation=cv2.INTER_CUBIC)

    model = load_pytorch_model()
    pytorch_sr = run_pytorch(model, lr)
    onnx_sr = run_onnx(lr)
    hailo_sr = run_hailo(lr)

    results = [
        ("LR input", lr_vis, None),
        ("Nearest", nearest, psnr(nearest, hr)),
        ("Bicubic", bicubic, psnr(bicubic, hr)),
        ("ESPCN PyTorch", pytorch_sr, psnr(pytorch_sr, hr)),
        ("ESPCN ONNX", onnx_sr, psnr(onnx_sr, hr)),
        ("ESPCN Hailo", hailo_sr, psnr(hailo_sr, hr)),
        ("Ground Truth", hr, None),
    ]

    panels = []
    for name, image, score in results:
        if score is None:
            label = name
        else:
            label = f"{name} | {score:.2f} dB"
        panels.append(add_label(image, label))

    row1 = np.hstack(panels[:4])      # LR, Nearest, Bicubic, GT
    row2 = np.hstack(panels[4:])      # PyTorch, ONNX, Hailo

    # aggiungo un pannello nero vuoto per avere la stessa larghezza
    blank = np.zeros_like(panels[0])
    row2 = np.hstack([row2, blank])

    comparison = np.vstack([row1, row2])
    cv2.imwrite(str(OUT_PATH), comparison)

    print(f"Input image: {INPUT_IMAGE}")
    for name, _, score in results:
        if score is not None:
            print(f"{name:14s}: {score:.3f} dB")

    print(f"Saved figure to: {OUT_PATH}")


if __name__ == "__main__":
    main()
