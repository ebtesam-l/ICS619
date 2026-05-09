# USAGE: python detect_align.py --input frames/ --output crops/
#
# Detects faces with MTCNN, aligns each face to a 512x512 canonical template
# using its 5 landmarks, and writes one PNG per detected face.
# A sidecar landmarks_512.npz is also written so downstream stages can reuse the points.

import argparse
import os
import glob
import numpy as np
import cv2

CROP_SIZE = 512  # 512x512 output per spec

# Standard ArcFace 5-point template (defined for 112x112), scaled to CROP_SIZE.
_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)
TEMPLATE = _TEMPLATE_112 * (CROP_SIZE / 112.0)


def _build_mtcnn():
    """Lazy import so users without facenet-pytorch can still import the file."""
    from facenet_pytorch import MTCNN
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MTCNN(keep_all=True, device=device)


def _align(img, landmarks5):
    """Similarity transform `img` so its 5 landmarks match TEMPLATE."""
    src = np.asarray(landmarks5, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR)


def _list_images(input_dir):
    """Case-insensitive listing of common image extensions."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    out = []
    for name in os.listdir(input_dir):
        if os.path.splitext(name)[1].lower() in exts:
            out.append(os.path.join(input_dir, name))
    return sorted(out)


def detect_and_align(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mtcnn = _build_mtcnn()

    image_paths = _list_images(input_dir)
    print(f"  scanning {len(image_paths)} images in {input_dir}")

    landmarks_log = {}
    written = 0
    n_no_face = 0
    n_unread = 0

    for p in image_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            n_unread += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, lmks = mtcnn.detect(rgb, landmarks=True)
        if lmks is None:
            n_no_face += 1
            continue

        base = os.path.splitext(os.path.basename(p))[0]
        for i, lm in enumerate(lmks):
            crop = _align(bgr, lm)
            if crop is None:
                continue
            out_name = f"{base}_face{i}.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, crop)
            landmarks_log[out_name] = lm.astype(np.float32)
            written += 1

    np.savez(os.path.join(output_dir, "landmarks_512.npz"), **landmarks_log)
    print(f"  no-face frames: {n_no_face}, unreadable: {n_unread}")
    return written


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", help="folder of frames", required=True)
    argParser.add_argument("--output", help="folder to write aligned crops", required=True)
    args = argParser.parse_args()

    n = detect_and_align(args.input, args.output)
    print(f"Aligned faces written: {n}")
