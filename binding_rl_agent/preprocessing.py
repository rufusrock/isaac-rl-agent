from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Raw frame utilities (input: HWC BGR from window capture)
# ---------------------------------------------------------------------------

def resize_frame(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)


def to_grayscale(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def to_equalized_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def to_bilateral_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def to_multichannel(frame_bgr: np.ndarray) -> np.ndarray:
    """Returns (2, H, W) uint8: [equalized_gray, bilateral_gray]."""
    eq = to_equalized_gray(frame_bgr)
    bil = to_bilateral_gray(frame_bgr)
    return np.stack([eq, bil], axis=0)


def to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Returns (3, H, W) uint8 in RGB channel order."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return np.moveaxis(rgb, -1, 0)  # (H,W,3) → (3,H,W)


def stack_frames(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("Expected at least one frame to stack.")
    stacked = np.stack(frames, axis=0)  # (N, H, W) or (N, C, H, W)
    if stacked.ndim == 4:
        # multichannel: (N, C, H, W) → (N*C, H, W)
        n, c, h, w = stacked.shape
        return stacked.reshape(n * c, h, w)
    return stacked


# ---------------------------------------------------------------------------
# Dataset frame transforms
#
# All take a single (3, H, W) uint8 RGB frame (as stored by IsaacRolloutDataset)
# and return a processed frame ready for stacking.
# ---------------------------------------------------------------------------

def _rgb_chw_to_bgr_hwc(frame: np.ndarray) -> np.ndarray:
    """(3, H, W) RGB uint8 → (H, W, 3) BGR uint8 for cv2 functions."""
    return np.ascontiguousarray(np.moveaxis(frame, 0, -1)[:, :, ::-1])


def frame_gray(frame_rgb_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) RGB → (H,W) uint8 grayscale."""
    return cv2.cvtColor(_rgb_chw_to_bgr_hwc(frame_rgb_chw), cv2.COLOR_BGR2GRAY)


def frame_eq_gray(frame_rgb_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) RGB → (H,W) uint8 equalized grayscale."""
    gray = cv2.cvtColor(_rgb_chw_to_bgr_hwc(frame_rgb_chw), cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def frame_multichannel(frame_rgb_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) RGB → (2,H,W) uint8 [equalized_gray, bilateral_gray]."""
    return to_multichannel(_rgb_chw_to_bgr_hwc(frame_rgb_chw))


def frame_rgb(frame_rgb_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) RGB → (3,H,W) RGB (identity)."""
    return frame_rgb_chw


def frame_hsv_sv(frame_rgb_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) RGB → (2,H,W) uint8 [saturation, value]."""
    hsv = cv2.cvtColor(_rgb_chw_to_bgr_hwc(frame_rgb_chw), cv2.COLOR_BGR2HSV)
    return np.stack([hsv[:, :, 1], hsv[:, :, 2]], axis=0)


def frame_mc_sat(frame_rgb_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) RGB → (3,H,W) uint8 [equalized_gray, bilateral_gray, saturation].

    Combines the structural contrast of multichannel with a saturation channel
    that highlights enemies/pickups (colorful) over background (desaturated).
    """
    bgr = _rgb_chw_to_bgr_hwc(frame_rgb_chw)
    eq = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    bil = cv2.bilateralFilter(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), d=9, sigmaColor=75, sigmaSpace=75)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    return np.stack([eq, bil, sat], axis=0)


# Callable lookup by name — used by TrainConfig.frame_mode
FRAME_TRANSFORMS: dict[str, callable] = {
    "gray":         frame_gray,
    "eq_gray":      frame_eq_gray,
    "multichannel": frame_multichannel,
    "rgb":          frame_rgb,
    "hsv_sv":       frame_hsv_sv,
    "mc_sat":       frame_mc_sat,
}


# ---------------------------------------------------------------------------
# Live-capture frame transforms (input: HWC BGR from window capture)
# Mirror of FRAME_TRANSFORMS but operating on raw BGR capture frames.
# ---------------------------------------------------------------------------

def to_eq_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Returns (H, W) uint8 equalized grayscale."""
    return cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))


def to_hsv_sv(frame_bgr: np.ndarray) -> np.ndarray:
    """Returns (2, H, W) uint8 [saturation, value]."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return np.stack([hsv[:, :, 1], hsv[:, :, 2]], axis=0)


def to_mc_sat(frame_bgr: np.ndarray) -> np.ndarray:
    """Returns (3, H, W) uint8 [equalized_gray, bilateral_gray, saturation]."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    bil = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    sat = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    return np.stack([eq, bil, sat], axis=0)


LIVE_FRAME_TRANSFORMS: dict[str, callable] = {
    "gray":         to_grayscale,
    "eq_gray":      to_eq_gray,
    "multichannel": to_multichannel,
    "rgb":          to_rgb,
    "hsv_sv":       to_hsv_sv,
    "mc_sat":       to_mc_sat,
}


def resize_frame_rgb(frame_rgb_chw: np.ndarray, size: int) -> np.ndarray:
    """Resize a (3,H,W) uint8 RGB frame to (3,size,size)."""
    hwc = np.moveaxis(frame_rgb_chw, 0, -1)  # (H,W,3)
    resized = cv2.resize(hwc, (size, size), interpolation=cv2.INTER_AREA)
    return np.moveaxis(resized, -1, 0)        # (3,size,size)
