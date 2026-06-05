"""
cv_engine.py — OpenCV preprocessing for medical report images.

Single capability:
  preprocess_report_image(image_bytes) → (cleaned_data_url, metrics)

Pipeline (classical computer vision, no ML):
  1. Decode → grayscale
  2. CLAHE contrast enhancement
  3. Adaptive Gaussian threshold (handles uneven lighting on paper)
  4. Deskew via Hough line transform
  5. Encode as PNG data URL for GPT-4o Vision

This makes scanned/photographed lab reports far more legible to a
downstream Vision model — straighter, higher-contrast, lighting-normalised.
"""
from __future__ import annotations

import base64
import math
from typing import Any

import cv2
import numpy as np


def preprocess_report_image(image_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """
    Clean a photo/scan of a paper medical report so a Vision model can read it.

    Parameters
    ----------
    image_bytes : raw bytes of the uploaded image (JPEG / PNG / etc.)

    Returns
    -------
    (cleaned_data_url, metrics)
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupt file")

    h0, w0 = img.shape[:2]

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: CLAHE — local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Step 3: Adaptive Gaussian threshold — robust to uneven lighting
    thresh = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=8,
    )

    # Step 4: Deskew using Hough lines
    cleaned, skew_deg = _deskew(thresh)

    # Step 5: Encode as PNG data URL
    cleaned_url = _encode_png_data_url(cleaned)

    metrics = {
        "size":              f"{w0} x {h0}",
        "original_contrast": round(float(np.std(gray)), 1),
        "enhanced_contrast": round(float(np.std(contrast)), 1),
        "skew_corrected":    round(skew_deg, 2),
    }

    return cleaned_url, metrics


def _deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate to correct small skews using Hough line transform."""
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180,
        threshold=100,
        minLineLength=img.shape[1] // 4,
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return img, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angles.append(math.degrees(math.atan2(y2 - y1, x2 - x1)))

    if not angles:
        return img, 0.0

    median_angle = float(np.median(angles))
    # Only correct small skews (avoid flipping portrait images)
    if abs(median_angle) > 15:
        return img, 0.0

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def _encode_png_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"
