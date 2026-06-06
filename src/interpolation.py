"""
Image resizing utilities used in the super-resolution benchmarks.

This module provides:
- downscaling to generate low-resolution inputs;
- nearest-neighbor upscaling baseline;
- bicubic upscaling baseline.

Nearest and bicubic are used as traditional reference methods
to compare the performance of deep learning models such as
ESPCN and NiNASR.
"""

import cv2


def downscale(img, scale):
    """
    Reduce image resolution by a given scale factor.

    INTER_AREA is generally recommended for image shrinking
    because it reduces aliasing artifacts and preserves image quality.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    scale : int
        Downscaling factor.

    Returns
    -------
    np.ndarray
        Low-resolution image.
    """

    h, w = img.shape[:2]

    return cv2.resize(
        img,
        (w // scale, h // scale),
        interpolation=cv2.INTER_AREA
    )


def upscale_nearest(img, size):
    """
    Upscale an image using nearest-neighbor interpolation.

    This is the simplest possible upscaling method:
    each output pixel copies the value of the nearest input pixel.

    Advantages:
    - extremely fast
    - minimal computational cost

    Disadvantages:
    - blocky appearance
    - visible jagged edges
    """

    return cv2.resize(
        img,
        size,
        interpolation=cv2.INTER_NEAREST
    )


def upscale_bicubic(img, size):
    """
    Upscale an image using bicubic interpolation.

    Bicubic interpolation estimates each output pixel from
    neighbouring pixels using cubic functions.

    Advantages:
    - smoother images
    - higher reconstruction quality than nearest-neighbor

    Disadvantages:
    - higher computational cost
    - still limited compared to learned super-resolution models
    """

    return cv2.resize(
        img,
        size,
        interpolation=cv2.INTER_CUBIC
    )