"""
Single-frame camera test with ESPCN ONNX in color.

Pipeline:
Camera Module 3
→ central square crop
→ convert RGB to YCrCb
→ downscale Y, Cr, Cb to 224x224
→ apply ESPCN ONNX x2 only on Y channel
→ upscale Cr and Cb with bicubic interpolation
→ recombine YCrCb
→ save RGB super-resolved output
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from picamera2 import Picamera2


ONNX_PATH = Path("weights/espcn_x2_dynamic.onnx")
OUTPUT_DIR = Path("data/outputs/camera_espcn_single_frame_color")

LR_SIZE = 224
SCALE = 2


def center_square_crop(image):
    """
    Extract the largest possible central square crop from an image.
    """

    h, w = image.shape[:2]
    side = min(h, w)

    x0 = (w - side) // 2
    y0 = (h - side) // 2

    return image[y0:y0 + side, x0:x0 + side]


def prepare_tensor(gray_lr):
    """
    Convert a grayscale LR image to ONNX input format.

    Input:
    - shape: [H, W]
    - range: [0, 255]

    Output:
    - shape: [1, 1, H, W]
    - range: [0, 1]
    - dtype: float32
    """

    tensor = gray_lr.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, np.newaxis, :, :]

    return tensor


def run_onnx(session, input_name, tensor):
    """
    Run ESPCN ONNX inference and return an uint8 image.
    """

    output = session.run(None, {input_name: tensor})[0]

    output = np.squeeze(output)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Camera capture
    # -----------------------------

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={
            "size": (640, 480),
            "format": "RGB888",
        }
    )

    picam2.configure(config)
    picam2.start()

    frame_rgb = picam2.capture_array()

    picam2.stop()

    # -----------------------------
    # Central square crop
    # -----------------------------

    crop_rgb = center_square_crop(frame_rgb)

    # -----------------------------
    # RGB → YCrCb
    # -----------------------------

    # Y contains luminance/detail information.
    # Cr and Cb contain chrominance/color information.
    crop_ycrcb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(crop_ycrcb)

    # -----------------------------
    # Create LR input
    # -----------------------------

    y_lr = cv2.resize(
        y,
        (LR_SIZE, LR_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    cr_lr = cv2.resize(
        cr,
        (LR_SIZE, LR_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    cb_lr = cv2.resize(
        cb,
        (LR_SIZE, LR_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # RGB version of the LR crop, saved only for visual comparison.
    lr_rgb = cv2.resize(
        crop_rgb,
        (LR_SIZE, LR_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # -----------------------------
    # ESPCN ONNX on luminance channel
    # -----------------------------

    tensor = prepare_tensor(y_lr)

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    y_sr = run_onnx(
        session,
        input_name,
        tensor,
    )

    # -----------------------------
    # Bicubic upscaling of color channels
    # -----------------------------

    sr_size = (
        LR_SIZE * SCALE,
        LR_SIZE * SCALE,
    )

    cr_sr = cv2.resize(
        cr_lr,
        sr_size,
        interpolation=cv2.INTER_CUBIC,
    )

    cb_sr = cv2.resize(
        cb_lr,
        sr_size,
        interpolation=cv2.INTER_CUBIC,
    )

    # -----------------------------
    # Reconstruct RGB image
    # -----------------------------

    sr_ycrcb = cv2.merge(
        [
            y_sr,
            cr_sr,
            cb_sr,
        ]
    )

    sr_rgb = cv2.cvtColor(
        sr_ycrcb,
        cv2.COLOR_YCrCb2RGB,
    )

    # -----------------------------
    # Save outputs
    # -----------------------------

    cv2.imwrite(str(OUTPUT_DIR / "frame_original.png"), frame_rgb)
    cv2.imwrite(str(OUTPUT_DIR / "crop_square_rgb.png"), crop_rgb)
    cv2.imwrite(str(OUTPUT_DIR / "crop_lr_224_rgb.png"), lr_rgb)
    cv2.imwrite(str(OUTPUT_DIR / "y_lr_224.png"), y_lr)
    cv2.imwrite(str(OUTPUT_DIR / "y_sr_448.png"), y_sr)
    cv2.imwrite(str(OUTPUT_DIR / "crop_sr_448_rgb.png"), sr_rgb)

    print("Saved outputs to:", OUTPUT_DIR)
    print("Frame shape:", frame_rgb.shape)
    print("Crop RGB shape:", crop_rgb.shape)
    print("LR RGB shape:", lr_rgb.shape)
    print("SR RGB shape:", sr_rgb.shape)


if __name__ == "__main__":
    main()