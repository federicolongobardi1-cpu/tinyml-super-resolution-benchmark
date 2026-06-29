"""
Capture one frame from Raspberry Pi Camera Module 3
and save the original frame and a central square crop.
"""

import cv2
from pathlib import Path
from picamera2 import Picamera2


OUTPUT_DIR = Path("data/outputs/camera_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def center_square_crop(image):
    h, w = image.shape[:2]
    side = min(h, w)

    x0 = (w - side) // 2
    y0 = (h - side) // 2

    return image[y0:y0 + side, x0:x0 + side]


def main():
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )

    picam2.configure(config)
    picam2.start()

    frame_rgb = picam2.capture_array()
    picam2.stop()

    crop_rgb = center_square_crop(frame_rgb)

    cv2.imwrite(str(OUTPUT_DIR / "frame_original.png"), frame_rgb)
    cv2.imwrite(str(OUTPUT_DIR / "crop_square.png"), crop_rgb)

    print("Saved:")
    print(OUTPUT_DIR / "frame_original.png")
    print(OUTPUT_DIR / "crop_square.png")
    print("Frame shape:", frame_rgb.shape)
    print("Crop shape:", crop_rgb.shape)


if __name__ == "__main__":
    main()