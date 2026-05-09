# USAGE: python degrade.py --dataset data/saudi_dataset
#
# Reads splits/train.txt and splits/test.txt under --dataset, then for each
# filename copies the HQ crop to <split>/HQ/<name> and writes a degraded LQ
# version to <split>/LQ/<name>.
#
# Degradation pipeline lifted from GFPGAN's FFHQDegradationDataset.__getitem__:
#     blur -> downsample -> Gaussian noise -> JPEG compression -> resize back.
# Hardcoded values match GFPGAN's train_gfpgan_v1_simple.yml.

import argparse
import math
import os
import shutil

import cv2
import numpy as np
from basicsr.data import degradations as deg

# --- Hardcoded GFPGAN-standard config ---------------------------------------
BLUR_KERNEL_SIZE = 41
KERNEL_LIST      = ["iso", "aniso"]
KERNEL_PROB      = [0.5, 0.5]
BLUR_SIGMA       = [0.1, 10.0]
DOWNSAMPLE_RANGE = [0.8, 8.0]
NOISE_RANGE      = [0.0, 20.0]
JPEG_RANGE       = [60, 100]

# Color jitter and grayscale are intentionally OFF by default.
COLOR_JITTER_PROB = None
GRAY_PROB         = None
# ----------------------------------------------------------------------------


def degrade_image(img_gt_uint8):
    """Apply the GFPGAN degradation pipeline to one image.

    Input/output are uint8 BGR numpy arrays of identical shape.
    """
    h, w, _ = img_gt_uint8.shape
    img_gt = img_gt_uint8.astype(np.float32) / 255.0  # [0, 1]

    # 1) Blur
    kernel = deg.random_mixed_kernels(
        KERNEL_LIST, KERNEL_PROB,
        BLUR_KERNEL_SIZE, BLUR_SIGMA, BLUR_SIGMA,
        [-math.pi, math.pi], noise_range=None,
    )
    img_lq = cv2.filter2D(img_gt, -1, kernel)

    # 2) Downsample
    scale = np.random.uniform(DOWNSAMPLE_RANGE[0], DOWNSAMPLE_RANGE[1])
    img_lq = cv2.resize(
        img_lq,
        (max(1, int(w // scale)), max(1, int(h // scale))),
        interpolation=cv2.INTER_LINEAR,
    )

    # 3) Gaussian noise
    if NOISE_RANGE is not None:
        img_lq = deg.random_add_gaussian_noise(img_lq, NOISE_RANGE)

    # 4) JPEG compression
    if JPEG_RANGE is not None:
        img_lq = deg.random_add_jpg_compression(img_lq, JPEG_RANGE)

    # 5) Resize back to original size
    img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

    # 6) Optional colour jitter (additive RGB shift, in [-shift, shift]/255)
    if COLOR_JITTER_PROB is not None and np.random.uniform() < COLOR_JITTER_PROB:
        shift = np.random.uniform(-20.0 / 255.0, 20.0 / 255.0, 3).astype(np.float32)
        img_lq = np.clip(img_lq + shift, 0.0, 1.0)

    # 7) Optional grayscale
    if GRAY_PROB is not None and np.random.uniform() < GRAY_PROB:
        gray = cv2.cvtColor((img_lq * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img_lq = (np.tile(gray[:, :, None], [1, 1, 3]).astype(np.float32) / 255.0)

    img_lq = np.clip(img_lq * 255.0, 0, 255).round().astype(np.uint8)
    return img_lq


def _process_split(dataset_root, split_name, filenames):
    src_dir = os.path.join(dataset_root, "images")
    hq_dir = os.path.join(dataset_root, split_name, "HQ")
    lq_dir = os.path.join(dataset_root, split_name, "LQ")
    os.makedirs(hq_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)

    n_done, n_missing = 0, 0
    for name in filenames:
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            n_missing += 1
            continue
        img = cv2.imread(src)
        if img is None:
            n_missing += 1
            continue

        # HQ: copy as-is. LQ: degraded version with same filename.
        shutil.copy2(src, os.path.join(hq_dir, name))
        cv2.imwrite(os.path.join(lq_dir, name), degrade_image(img))
        n_done += 1

    return n_done, n_missing


def degrade_dataset(dataset_root):
    splits_dir = os.path.join(dataset_root, "splits")
    if not os.path.isdir(splits_dir):
        raise FileNotFoundError(
            f"{splits_dir} not found. Run preprocess/split.py first."
        )

    summary = {}
    for split_name in ("train", "test"):
        split_file = os.path.join(splits_dir, f"{split_name}.txt")
        if not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        n_done, n_missing = _process_split(dataset_root, split_name, names)
        summary[split_name] = (n_done, n_missing)
        print(f"  {split_name}: wrote {n_done} HQ/LQ pairs, missing {n_missing}")
    return summary


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--dataset",
        help="dataset root containing images/ and splits/",
        required=True,
    )
    args = argParser.parse_args()

    degrade_dataset(args.dataset)
