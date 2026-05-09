# USAGE: python filter_identity.py --input crops/ --output crops_final/ --model checkpoints/resnet18_110.pth
#
# 1) Embeds each crop with the same ArcFace ResNet18 used by IDS.py.
# 2) Groups crops into identities by greedy cosine-similarity clustering (>= IDENTITY_SIM).
# 3) Within each identity, drops near-duplicate crops (cosine >= CONTENT_SIM).
# 4) Caps each identity at MAX_PER_ID crops.
#
# Writes a manifest identities.csv mapping kept_crop -> identity_id, and copies kept crops to output_dir.

import argparse
import os
import glob
import shutil
import sys
import csv
import numpy as np
import cv2
import torch
from torch.nn import DataParallel

# Reuse the user's ArcFace network defined in src/metrics/IDS.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_METRICS = os.path.join(_HERE, "..", "metrics")
sys.path.insert(0, _METRICS)
from IDS import resnet_face18, load_image, Config  # noqa: E402

IDENTITY_SIM = 0.70   # >= this cosine ==> same identity
CONTENT_SIM  = 0.95   # >= this cosine ==> near-duplicate within an identity
MAX_PER_ID   = 5


def _load_arcface(model_path):
    opt = Config()
    model = resnet_face18(opt.use_se)
    model = DataParallel(model)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device


def _embed(model, device, img_path):
    t = load_image(img_path)              # 1x128x128 grayscale, normalized
    if t is None:
        return None
    with torch.no_grad():
        e = model(t.unsqueeze(0).to(device)).cpu().numpy().squeeze()
    n = np.linalg.norm(e)
    return e / n if n > 0 else None


def _cosine(a, b):
    return float(np.dot(a, b))


def filter_identity(input_dir, output_dir, model_path):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.jpeg"))
    )

    model, device = _load_arcface(model_path)

    # 1) Embed everything.
    embeddings = []
    for p in paths:
        e = _embed(model, device, p)
        if e is not None:
            embeddings.append((p, e))

    # 2) Greedy cluster into identities.
    identity_centroids = []   # list of (centroid_embedding, [member_paths_with_emb])
    for path, emb in embeddings:
        assigned = False
        for i, (cen, members) in enumerate(identity_centroids):
            if _cosine(emb, cen) >= IDENTITY_SIM:
                members.append((path, emb))
                # update centroid as running mean (re-normalised)
                m = np.mean([e for _, e in members], axis=0)
                m /= (np.linalg.norm(m) or 1.0)
                identity_centroids[i] = (m, members)
                assigned = True
                break
        if not assigned:
            identity_centroids.append((emb, [(path, emb)]))

    # 3) For each identity, drop near-duplicate frames; 4) cap at MAX_PER_ID.
    kept_rows = []
    for ident_id, (_, members) in enumerate(identity_centroids):
        unique_members = []
        for path, emb in members:
            if any(_cosine(emb, e2) >= CONTENT_SIM for _, e2 in unique_members):
                continue
            unique_members.append((path, emb))
            if len(unique_members) >= MAX_PER_ID:
                break
        for path, _ in unique_members:
            name = os.path.basename(path)
            shutil.copy2(path, os.path.join(output_dir, name))
            kept_rows.append((name, ident_id))

    # Manifest.
    manifest = os.path.join(output_dir, "identities.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "identity_id"])
        w.writerows(kept_rows)

    return len(kept_rows), len(identity_centroids)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", help="folder of frontal crops", required=True)
    argParser.add_argument("--output", help="folder to write final crops", required=True)
    argParser.add_argument("--model", help="path to ArcFace resnet18_110.pth", required=True)
    args = argParser.parse_args()

    n_kept, n_id = filter_identity(args.input, args.output, args.model)
    print(f"kept {n_kept} crops across {n_id} identities (max {MAX_PER_ID} per id)")
