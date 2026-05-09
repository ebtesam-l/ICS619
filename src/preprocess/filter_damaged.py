# USAGE: python filter_damaged.py --input crops/ --output crops_clean/
#
# Drops images that are NaN/empty or too blurry.
# Blur is measured by the variance of the Laplacian; very low variance ==> blurry / featureless.

import argparse
import os
import glob
import shutil
import numpy as np
import cv2

LAPLACIAN_VAR_MIN = 100.0  # crops with var < this are rejected as blurry / damaged


def is_damaged(img):
    """Return True if `img` should be rejected."""
    if img is None or img.size == 0:
        return True
    if not np.isfinite(img).all():
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if not np.isfinite(var):
        return True
    return var < LAPLACIAN_VAR_MIN


def filter_damaged(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.jpeg"))
    )

    kept, dropped = 0, 0
    for p in paths:
        img = cv2.imread(p)
        if is_damaged(img):
            dropped += 1
            continue
        shutil.copy2(p, os.path.join(output_dir, os.path.basename(p)))
        kept += 1

    return kept, dropped


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", help="folder of aligned crops", required=True)
    argParser.add_argument("--output", help="folder to write kept crops", required=True)
    args = argParser.parse_args()

    kept, dropped = filter_damaged(args.input, args.output)
    print(f"kept {kept} | dropped {dropped}")
