# USAGE: python split.py --input crops_final/ --output data/saudi_dataset/
#
# Reads identities.csv from --input, splits identities (not crops) into train/test
# at TEST_RATIO with a fixed seed, copies crops to <output>/images/, and writes
# splits/train.txt and splits/test.txt with one filename per line.

import argparse
import os
import glob
import shutil
import csv
import random

TEST_RATIO = 0.20
SEED = 42


def split_train_test(input_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "splits"), exist_ok=True)

    manifest = os.path.join(input_dir, "identities.csv")
    if not os.path.exists(manifest):
        raise FileNotFoundError(
            f"{manifest} not found. Run filter_identity.py first."
        )

    # Read manifest -> dict[identity_id] = [filename, ...]
    by_id = {}
    with open(manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_id.setdefault(int(row["identity_id"]), []).append(row["filename"])

    # Split identities (so train/test never share a face).
    ids = sorted(by_id)
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * TEST_RATIO))
    test_ids = set(ids[:n_test])

    train_lines, test_lines = [], []
    for ident_id, names in by_id.items():
        target = test_lines if ident_id in test_ids else train_lines
        for n in names:
            shutil.copy2(os.path.join(input_dir, n),
                         os.path.join(output_dir, "images", n))
            target.append(n)

    with open(os.path.join(output_dir, "splits", "train.txt"), "w") as f:
        f.write("\n".join(sorted(train_lines)) + "\n")
    with open(os.path.join(output_dir, "splits", "test.txt"), "w") as f:
        f.write("\n".join(sorted(test_lines)) + "\n")

    return len(train_lines), len(test_lines), len(ids), len(test_ids)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", help="folder with identities.csv", required=True)
    argParser.add_argument("--output", help="dataset root (writes images/ and splits/)", required=True)
    args = argParser.parse_args()

    n_train, n_test, n_id, n_test_id = split_train_test(args.input, args.output)
    print(f"identities: {n_id} total, {n_test_id} held out for test")
    print(f"crops:      {n_train} train, {n_test} test")
