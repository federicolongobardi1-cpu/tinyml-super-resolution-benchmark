"""
Image quality metrics used in the super-resolution benchmarks.

This module currently provides:
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)

PSNR is the main reconstruction metric used to compare:
- nearest-neighbor interpolation
- bicubic interpolation
- deep learning models (ESPCN, NiNASR)

Higher PSNR values indicate better reconstruction quality.
"""

import numpy as np


def mse(a, b):
    """
    Compute the Mean Squared Error (MSE) between two images.

    Parameters
    ----------
    a : np.ndarray
        Reference image.
    b : np.ndarray
        Reconstructed image.

    Returns
    -------
    float
        Average squared pixel error.
    """

    return np.mean(
        (a.astype(np.float32) - b.astype(np.float32)) ** 2
    )


def psnr(a, b):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR).

    PSNR is derived from the MSE and is expressed in decibels (dB).

    Typical interpretation:
    - < 20 dB : poor reconstruction
    - 20–30 dB : acceptable reconstruction
    - 30–40 dB : good reconstruction
    - > 40 dB : very high similarity

    Parameters
    ----------
    a : np.ndarray
        Reference image.
    b : np.ndarray
        Reconstructed image.

    Returns
    -------
    float
        PSNR value in dB.
    """

    error = mse(a, b)

    # Perfect reconstruction.
    if error < 1e-12:
        return float("inf")

    # Images are stored as uint8, therefore MAX_I = 255.
    return 10 * np.log10((255**2) / error)

def print_image_stats(name, image):
    print(
        f"{name}: "
        f"shape={image.shape}, "
        f"dtype={image.dtype}, "
        f"min={image.min()}, "
        f"max={image.max()}"
    )