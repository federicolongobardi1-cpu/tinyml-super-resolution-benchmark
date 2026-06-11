"""
Record a short comparison video from Camera Module 3 using ESPCN ONNX.

Output video:
Left  = low-resolution crop enlarged with nearest-neighbor
Right = ESPCN super-resolution output

Color handling note:
In this Raspberry Pi + Picamera2 setup, frames are captured as RGB888.
The previous RGB→BGR conversion caused corrupted colors, so frames are
written directly to VideoWriter.
"""

import cv2
import time
import psutil
import numpy as np
import onnxruntime as ort
from pathlib import Path
from picamera2 import Picamera2


ONNX_PATH = Path("weights/espcn_x2_dynamic.onnx")

OUTPUT_DIR = Path("data/outputs/camera_espcn_video")
OUTPUT_VIDEO = OUTPUT_DIR / "lr_vs_sr.mp4"

CAMERA_SIZE = (640, 480)
LR_SIZE = 224
SCALE = 2

DURATION_SECONDS = 20
OUTPUT_FPS = 15


def center_square_crop(image):
    """
    Extract the largest possible central square crop.
    """

    h, w = image.shape[:2]
    side = min(h, w)

    x0 = (w - side) // 2
    y0 = (h - side) // 2

    return image[y0:y0 + side, x0:x0 + side]


def prepare_tensor(gray_lr):
    """
    Convert a grayscale LR image to ONNX input format:
    [H, W] uint8 -> [1, 1, H, W] float32 in [0, 1].
    """

    tensor = gray_lr.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, np.newaxis, :, :]

    return tensor


def run_onnx(session, input_name, tensor):
    """
    Run ESPCN ONNX inference on the luminance channel.
    """

    output = session.run(None, {input_name: tensor})[0]

    output = np.squeeze(output)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    return output


def super_resolve_color(session, input_name, crop_rgb):
    """
    Apply ESPCN only to the Y luminance channel.

    RGB crop
    -> YCrCb
    -> ESPCN on Y
    -> bicubic upscale on Cr/Cb
    -> RGB reconstruction
    """

    crop_ycrcb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(crop_ycrcb)

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

    tensor = prepare_tensor(y_lr)
    y_sr = run_onnx(session, input_name, tensor)

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

    sr_ycrcb = cv2.merge([y_sr, cr_sr, cb_sr])
    sr_rgb = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2RGB)

    lr_rgb = cv2.resize(
        crop_rgb,
        (LR_SIZE, LR_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    lr_display = cv2.resize(
        lr_rgb,
        sr_size,
        interpolation=cv2.INTER_NEAREST,
    )

    return lr_display, sr_rgb


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={
            "size": CAMERA_SIZE,
            "format": "RGB888",
        }
    )

    picam2.configure(config)
    picam2.start()

    output_width = LR_SIZE * SCALE * 2
    output_height = LR_SIZE * SCALE

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (output_width, output_height),
    )

    if not writer.isOpened():
        picam2.stop()
        raise RuntimeError("Could not open VideoWriter.")

    process = psutil.Process()

    start_time = time.perf_counter()
    last_print_time = start_time
    frame_count = 0

    print("Recording video. Press Ctrl+C to stop.")
    print(f"Output: {OUTPUT_VIDEO}")

    try:
        while True:
            now = time.perf_counter()

            if now - start_time >= DURATION_SECONDS:
                break

            frame_rgb = picam2.capture_array()

            crop_rgb = center_square_crop(frame_rgb)

            lr_display, sr_rgb = super_resolve_color(
                session,
                input_name,
                crop_rgb,
            )

            combined_rgb = np.hstack(
                [
                    lr_display,
                    sr_rgb,
                ]
            )

            # In this setup, writing RGB directly gives correct colors.
            # Do not convert RGB to BGR here.
            writer.write(combined_rgb)

            frame_count += 1

            if now - last_print_time >= 1.0:
                elapsed = now - start_time
                fps = frame_count / elapsed

                cpu = psutil.cpu_percent(interval=None)
                ram = process.memory_info().rss / (1024 ** 2)

                print(
                    f"FPS: {fps:.2f} | "
                    f"CPU: {cpu:.1f}% | "
                    f"RAM: {ram:.1f} MB"
                )

                last_print_time = now

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        picam2.stop()
        writer.release()

    total_time = time.perf_counter() - start_time
    avg_fps = frame_count / total_time

    print("\nRecording completed.")
    print(f"Saved video to: {OUTPUT_VIDEO}")
    print(f"Frames: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()