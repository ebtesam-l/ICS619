# USAGE: python filter_pose.py --input crops/ --output crops_frontal/
#
# Rejects faces whose absolute yaw exceeds YAW_THRESHOLD_DEG, using the 5 landmarks
# saved by detect_align.py in landmarks_512.npz.

import argparse
import os
import glob
import shutil
import numpy as np

YAW_THRESHOLD_DEG = 45.0


def estimate_yaw_from_landmarks(landmarks5):
    """Rough yaw estimate from 5-point landmarks.

    Compares horizontal distance from the nose to the left vs right eye:
        ratio = (nose_x - leye_x) / (reye_x - leye_x)
    For a frontal face this is ~0.5; deviation maps to yaw via arcsin.
    """
    leye, reye, nose, *_ = np.asarray(landmarks5, dtype=np.float32)
    eye_dx = reye[0] - leye[0]
    if eye_dx <= 1e-6:
        return 90.0
    ratio = (nose[0] - leye[0]) / eye_dx
    delta = (ratio - 0.5) * 2.0   # in [-1, 1] for a typical face
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(np.degrees(np.arcsin(delta)))


def filter_pose(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(input_dir, "landmarks_512.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found. Run detect_align.py first so landmarks are available."
        )
    lmks = np.load(npz_path)

    paths = sorted(
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.jpeg"))
    )

    kept, dropped = 0, 0
    kept_landmarks = {}
    for p in paths:
        name = os.path.basename(p)
        if name not in lmks:
            dropped += 1
            continue
        yaw = estimate_yaw_from_landmarks(lmks[name])
        if abs(yaw) > YAW_THRESHOLD_DEG:
            dropped += 1
            continue
        shutil.copy2(p, os.path.join(output_dir, name))
        kept_landmarks[name] = lmks[name]
        kept += 1

    np.savez(os.path.join(output_dir, "landmarks_512.npz"), **kept_landmarks)
    return kept, dropped


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", help="folder of crops with landmarks_512.npz", required=True)
    argParser.add_argument("--output", help="folder to write frontal crops", required=True)
    args = argParser.parse_args()

    kept, dropped = filter_pose(args.input, args.output)
    print(f"kept {kept} | dropped {dropped}")
