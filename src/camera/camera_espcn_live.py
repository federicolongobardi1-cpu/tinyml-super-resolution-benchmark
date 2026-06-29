"""
Live camera benchmark with ESPCN ONNX.

Pipeline:
Camera Module 3
→ central square crop
→ downscale to 224x224
→ ESPCN ONNX on Y channel
→ bicubic upscale on color channels
→ FPS / CPU / RAM measurement
"""

import cv2
import time
import psutil
import numpy as np
import onnxruntime as ort
from pathlib import Path
from picamera2 import Picamera2


ONNX_PATH = Path("weights/espcn_x2_dynamic.onnx")

LR_SIZE = 224
SCALE = 2
CAMERA_SIZE = (640, 480)

PRINT_INTERVAL = 1.0


def center_square_crop(image):
    h, w = image.shape[:2]
    side = min(h, w)

    x0 = (w - side) // 2
    y0 = (h - side) // 2

    return image[y0:y0 + side, x0:x0 + side]


def prepare_tensor(gray_lr):
    tensor = gray_lr.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, np.newaxis, :, :]
    return tensor


def run_onnx(session, input_name, tensor):
    output = session.run(None, {input_name: tensor})[0]
    output = np.squeeze(output)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    return output


def super_resolve_color(session, input_name, crop_rgb):
    crop_ycrcb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(crop_ycrcb)

    y_lr = cv2.resize(y, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_AREA)
    cr_lr = cv2.resize(cr, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_AREA)
    cb_lr = cv2.resize(cb, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_AREA)

    tensor = prepare_tensor(y_lr)
    y_sr = run_onnx(session, input_name, tensor)

    sr_size = (LR_SIZE * SCALE, LR_SIZE * SCALE)

    cr_sr = cv2.resize(cr_lr, sr_size, interpolation=cv2.INTER_CUBIC)
    cb_sr = cv2.resize(cb_lr, sr_size, interpolation=cv2.INTER_CUBIC)

    sr_ycrcb = cv2.merge([y_sr, cr_sr, cb_sr])
    sr_rgb = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2RGB)

    return sr_rgb


def main():
    process = psutil.Process()

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

    frame_count = 0
    start_time = time.perf_counter()
    last_print_time = start_time

    print("Live benchmark started. Press Ctrl+C to stop.")

    try:
        while True:
            frame_rgb = picam2.capture_array()

            crop_rgb = center_square_crop(frame_rgb)

            _ = super_resolve_color(
                session,
                input_name,
                crop_rgb,
            )

            frame_count += 1
            now = time.perf_counter()

            if now - last_print_time >= PRINT_INTERVAL:
                elapsed = now - start_time
                fps = frame_count / elapsed

                cpu_percent = psutil.cpu_percent(interval=None)
                ram_mb = process.memory_info().rss / (1024 ** 2)

                print(
                    f"FPS: {fps:.2f} | "
                    f"CPU: {cpu_percent:.1f}% | "
                    f"RAM: {ram_mb:.1f} MB"
                )

                last_print_time = now

    except KeyboardInterrupt:
        print("\nStopping live benchmark...")

    finally:
        picam2.stop()


if __name__ == "__main__":
    main()