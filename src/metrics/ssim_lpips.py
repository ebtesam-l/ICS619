import os
import cv2
import numpy as np
import torch
from basicsr.metrics.psnr_ssim import calculate_ssim
from basicsr.utils import img2tensor
import lpips
from torchvision.transforms.functional import normalize

def compute_ssim(gt_folder, restored_folder, crop_border=0, input_order='HWC', test_y_channel=False):
    """
    Compute SSIM for all image pairs in the given folders.
    """
    gt_images = sorted(os.listdir(gt_folder))
    restored_images = sorted(os.listdir(restored_folder))

    ssim_scores = []

    for gt_img_name, restored_img_name in zip(gt_images, restored_images):
        gt_path = os.path.join(gt_folder, gt_img_name)
        restored_path = os.path.join(restored_folder, restored_img_name)

        # Read images
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        restored_img = cv2.imread(restored_path, cv2.IMREAD_COLOR)

        if gt_img is None or restored_img is None:
            print(f"Error reading images: {gt_img_name}, {restored_img_name}")
            continue

        # Ensure the images have the same dimensions
        if gt_img.shape != restored_img.shape:
            print(f"Image size mismatch: {gt_img_name}, {restored_img_name}")
            continue

        # Convert images to float32
        gt_img = gt_img.astype(np.float32)
        restored_img = restored_img.astype(np.float32)

        # Compute SSIM
        ssim = calculate_ssim(gt_img, restored_img, crop_border=crop_border, input_order=input_order, test_y_channel=test_y_channel)
        ssim_scores.append(ssim)

        #print(f"{gt_img_name} - SSIM: {ssim:.4f}")

    # Compute average SSIM
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    print(f"\nAverage SSIM: {avg_ssim:.4f}")
    return avg_ssim

def compute_lpips(gt_folder, restored_folder, net_type='vgg'):
    """
    Compute LPIPS for all image pairs in the given folders.
    """
    gt_images = sorted(os.listdir(gt_folder))
    restored_images = sorted(os.listdir(restored_folder))

    lpips_scores = []

    # Initialize the LPIPS model
    loss_fn = lpips.LPIPS(net=net_type)
    loss_fn.eval()

    for gt_img_name, restored_img_name in zip(gt_images, restored_images):
        gt_path = os.path.join(gt_folder, gt_img_name)
        restored_path = os.path.join(restored_folder, restored_img_name)

        # Read images
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        restored_img = cv2.imread(restored_path, cv2.IMREAD_COLOR)

        if gt_img is None or restored_img is None:
            print(f"Error reading images: {gt_img_name}, {restored_img_name}")
            continue

        # Ensure the images have the same dimensions
        if gt_img.shape != restored_img.shape:
            print(f"Image size mismatch: {gt_img_name}, {restored_img_name}")
            continue

        # Convert BGR to RGB
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

        # Convert images to float32 and scale to [0, 1]
        gt_img = gt_img.astype(np.float32) / 255.0
        restored_img = restored_img.astype(np.float32) / 255.0

        # Convert to tensors and normalize to [-1, 1]
        gt_tensor = img2tensor(gt_img, bgr2rgb=False, float32=True)
        restored_tensor = img2tensor(restored_img, bgr2rgb=False, float32=True)

        normalize(gt_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        normalize(restored_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # Add batch dimension
        gt_tensor = gt_tensor.unsqueeze(0)
        restored_tensor = restored_tensor.unsqueeze(0)

        with torch.no_grad():
            lpips_value = loss_fn(gt_tensor, restored_tensor).item()

        lpips_scores.append(lpips_value)
        print(f"{gt_img_name} - LPIPS: {lpips_value:.4f}")

    # Compute average LPIPS
    avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
    print(f"\nAverage LPIPS: {avg_lpips:.4f}")
    return avg_lpips

# Example usage:
if __name__ == "__main__":
    gt_folder = './DifFace/CelebA-Test2/hq'
    restored_folder = './DifFace/outputlq256/restored_faces'

    print("Computing SSIM...")
    ssim = compute_ssim(gt_folder, restored_folder)

    print("\nComputing LPIPS...")
    lpips_ = compute_lpips(gt_folder, restored_folder)
