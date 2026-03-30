from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def _safe_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


@dataclass
class OCRPreprocessor:
    target_height: int = 64
    min_width: int = 64
    max_width: int = 1024
    apply_threshold: bool = True

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return _safe_uint8(image)
        return cv2.cvtColor(_safe_uint8(image), cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

    def deskew(self, image: np.ndarray) -> np.ndarray:
        inverted = cv2.bitwise_not(image)
        coords = np.column_stack(np.where(inverted > 0))
        if coords.size == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.1:
            return image

        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def resize_keep_ratio(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Image has invalid dimensions.")

        scale = self.target_height / float(height)
        resized_width = int(round(width * scale))
        resized_width = max(self.min_width, min(self.max_width, resized_width))
        resized = cv2.resize(image, (resized_width, self.target_height), interpolation=cv2.INTER_AREA)
        return resized, resized_width

    def normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        return (image - 0.5) / 0.5

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        gray = self.to_grayscale(image)
        blurred = self.gaussian_blur(gray)
        thresholded = self.adaptive_threshold(blurred) if self.apply_threshold else blurred
        deskewed = self.deskew(thresholded)
        resized, width = self.resize_keep_ratio(deskewed)
        normalized = self.normalize(resized)
        return normalized, width


class OCRAugmenter:
    def __init__(self, max_rotation: float = 2.0, noise_std: float = 8.0, blur_probability: float = 0.3):
        self.max_rotation = max_rotation
        self.noise_std = noise_std
        self.blur_probability = blur_probability

    def rotate(self, image: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-self.max_rotation, self.max_rotation)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return _safe_uint8(noisy)

    def blur(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.blur_probability:
            return image
        kernel_size = int(np.random.choice([3, 5]))
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = self.rotate(image)
        image = self.add_noise(image)
        image = self.blur(image)
        return image
