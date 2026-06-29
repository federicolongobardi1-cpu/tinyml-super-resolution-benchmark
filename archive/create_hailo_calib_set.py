import cv2
import numpy as np
from pathlib import Path


INPUT_DIR = Path("data/raw/DIV2K_valid_HR")
OUTPUT_PATH = Path("data/calibration/espcn_calib_1024.npy")

LR_SIZE = 224
N_SAMPLES = 1024
RANDOM_SEED = 42


def random_crop(gray, crop_size, rng):
    h, w = gray.shape

    if h < crop_size or w < crop_size:
        gray = cv2.resize(gray, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        return gray

    x0 = rng.integers(0, w - crop_size + 1)
    y0 = rng.integers(0, h - crop_size + 1)

    return gray[y0:y0 + crop_size, x0:x0 + crop_size]


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(INPUT_DIR.glob("*.png")))

    if len(image_paths) == 0:
        raise RuntimeError(f"No PNG images found in {INPUT_DIR}")

    samples = []

    for i in range(N_SAMPLES):
        path = image_paths[i % len(image_paths)]

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            print(f"Skipping unreadable image: {path}")
            continue

        crop = random_crop(gray, LR_SIZE, rng)

        crop = crop.astype(np.float32) / 255.0

        # Hailo expects NHWC calibration data.
        crop = crop[:, :, np.newaxis]

        samples.append(crop)

    calib = np.stack(samples, axis=0)

    np.save(OUTPUT_PATH, calib)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", calib.shape)
    print("Range:", calib.min(), calib.max())
    print("Number of source images:", len(image_paths))


if __name__ == "__main__":
    main()