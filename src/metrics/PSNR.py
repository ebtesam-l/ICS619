import os
import cv2
import numpy as np

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0 if img1.dtype == np.uint8 else 1.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def evaluate_psnr(dir_gt, dir_restored):
    psnr_scores = []
    filenames = sorted(os.listdir(dir_gt))

    for fname in filenames:
        path_gt = os.path.join(dir_gt, fname)
        path_restored = os.path.join(dir_restored, fname)

        if not os.path.exists(path_restored):
            print(f"[WARNING] Restored image not found for: {fname}")
            continue

        img_gt = cv2.imread(path_gt)
        img_restored = cv2.imread(path_restored)

        if img_gt is None or img_restored is None:
            print(f"[ERROR] Unable to read image pair: {fname}")
            continue

        if img_gt.shape != img_restored.shape:
            print(f"[SKIPPED] Shape mismatch for {fname} - GT: {img_gt.shape}, Restored: {img_restored.shape}")
            continue

        psnr = calculate_psnr(img_gt, img_restored)
        psnr_scores.append(psnr)

    if len(psnr_scores) == 0:
        print("No valid image pairs found.")
        return 0.0

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    print(f"Average PSNR over {len(psnr_scores)} images: {avg_psnr:.2f} dB")
    return avg_psnr
