# USAGE: python preprocess.py --input data/raw_videos --output data/saudi_dataset --arcface checkpoints/resnet18_110.pth
#
# End-to-end preprocessing: video frames -> aligned crops -> filtered crops -> identity-split dataset.
# Mirrors the structure of evaluate.py: this file is the entry point and imports each stage from its own file.

import argparse
import os
import sys

# Make sibling stage modules importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_frames import extract_frames           # noqa: E402
from detect_align import detect_and_align           # noqa: E402
from filter_damaged import filter_damaged           # noqa: E402
from filter_pose import filter_pose                 # noqa: E402
from filter_identity import filter_identity         # noqa: E402
from split import split_train_test                  # noqa: E402
from degrade import degrade_dataset                 # noqa: E402


def run(input_videos, output_root, arcface_path, work_dir=None):
    work_dir = work_dir or os.path.join(output_root, "_work")
    os.makedirs(work_dir, exist_ok=True)

    frames_dir       = os.path.join(work_dir, "01_frames")
    aligned_dir      = os.path.join(work_dir, "02_aligned")
    notdamaged_dir   = os.path.join(work_dir, "03_notdamaged")
    frontal_dir      = os.path.join(work_dir, "04_frontal")
    final_dir        = os.path.join(work_dir, "05_identity")

    # NOTE: filter_damaged needs landmarks, so we copy the .npz forward at each step.
    # extract_frames -> detect_align -> filter_damaged -> filter_pose -> filter_identity -> split

    print("[1/7] extract_frames...")
    n_frames = extract_frames(input_videos, frames_dir)
    print(f"      frames: {n_frames}")

    print("[2/7] detect_and_align (MTCNN)...")
    n_aligned = detect_and_align(frames_dir, aligned_dir)
    print(f"      aligned: {n_aligned}")

    print("[3/7] filter_damaged (Laplacian variance + NaN check)...")
    kept, dropped = filter_damaged(aligned_dir, notdamaged_dir)
    # carry landmarks forward (filter_damaged just copies images; refresh sidecar manually)
    _carry_landmarks(aligned_dir, notdamaged_dir)
    print(f"      kept {kept}, dropped {dropped}")

    print("[4/7] filter_pose (yaw threshold)...")
    kept, dropped = filter_pose(notdamaged_dir, frontal_dir)
    print(f"      kept {kept}, dropped {dropped}")

    print("[5/7] filter_identity (ArcFace embed, dedup, cap)...")
    n_kept, n_ids = filter_identity(frontal_dir, final_dir, arcface_path)
    print(f"      identities: {n_ids}, crops kept: {n_kept}")

    print("[6/7] split_train_test (identity-disjoint)...")
    n_train, n_test, n_id, n_test_id = split_train_test(final_dir, output_root)
    print(f"      identities: {n_id} ({n_test_id} test)")
    print(f"      crops: train={n_train}, test={n_test}")

    print("[7/7] degrade (GFPGAN pipeline -> HQ/LQ pairs)...")
    degrade_dataset(output_root)

    print(f"\nDone. Dataset written to: {output_root}")
    print("Layout:")
    print(f"  {output_root}/train/HQ, {output_root}/train/LQ")
    print(f"  {output_root}/test/HQ,  {output_root}/test/LQ")


def _carry_landmarks(src_dir, dst_dir):
    """filter_damaged.py only handles images; copy landmarks_512.npz forward, restricted to surviving names."""
    import numpy as np
    src = os.path.join(src_dir, "landmarks_512.npz")
    if not os.path.exists(src):
        return
    survivors = set(os.listdir(dst_dir))
    lmks = np.load(src)
    kept = {n: lmks[n] for n in lmks.files if n in survivors}
    np.savez(os.path.join(dst_dir, "landmarks_512.npz"), **kept)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", help="folder of videos (or a single video file)", required=True)
    argParser.add_argument("--output", help="dataset root to write", required=True)
    argParser.add_argument("--arcface", help="path to ArcFace resnet18_110.pth", required=True)
    argParser.add_argument("--work", help="intermediate working dir (defaults to <output>/_work)", default=None)
    args = argParser.parse_args()

    run(args.input, args.output, args.arcface, args.work)
