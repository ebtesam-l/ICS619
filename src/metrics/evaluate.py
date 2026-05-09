# USAGE: python src/metrics/evaluate.py --dirGT yourgroundtruth/ --dirEI yourenhancedimages/ --imagesize yourimagesize

import argparse
import os
import sys

# Make sibling metric modules importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import piq
from skimage.io import imread
import glob

from niqe import calculate_images_niqe
from IDS import calculate_cos_dist
from FID import calculate_fid
from PSNR import calculate_images_PSNR
from ssim_lpips import compute_ssim, compute_lpips
from PSNR import evaluate_psnr   # NOTE: was `from psnr import ...` — fixed for case-sensitive filesystems


# model_path_ids = 'resnet18_110.pth'

# download arcface model_path = 'resnet18_110.pth'
# https://drive.google.com/drive/folders/1k3RCSliF6PsujCMIdCD1hNM63EozlDIZ


if __name__ == "__main__":

    # torch.cuda.set_device(1)
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--imagesize", help="size of images, example: 128, 512, 1024", required=True, type=int)
    argParser.add_argument("--dirGT", help="directory containing ground truth images, example: celeba_train/", default=None)
    argParser.add_argument("--dirEI", help="directory containing enhanced images, example: celeba_train/", required=True)
    args = argParser.parse_args()

    dirGT = args.dirGT      # '.DifFace/CelebA-Test2/hq'
    dirEI = args.dirEI      # './DifFace/outputlq256/restored_faces'
    imagesize = args.imagesize  # 512

    print("Calculating NIQE...")
    thisNIQE = calculate_images_niqe(dirEI)
    print("NIQE | " + str(thisNIQE))

    if (dirGT is None):
        print("No ground truth dir provided, cannot calculate PSNR, IDS, or FID")
        exit()
    print("Calculating PSNR...")
    # thisPSNR = calculate_images_PSNR(dirGT,dirEI, imagesize)
    thisPSNR = evaluate_psnr(dirGT, dirEI)
    print("c | " + str(thisPSNR))

    # print("Calculating IDS...")
    # thisIDS = calculate_cos_dist(dirEI, dirGT, model_path_ids)
    # print("IDS | " + str(thisIDS))

    print("Calculating FID...")
    thisFID = calculate_fid(dirEI, dirGT)
    print("FID | " + str(thisFID))

    print("Calculating SSIM...")
    thisSSIM = compute_ssim(dirGT, dirEI)
    print("SSIM | " + str(thisSSIM))

    print("Calculating LPIPS...")
    thisLPIPS = compute_lpips(dirGT, dirEI)
    print("LPIPS | " + str(thisLPIPS))

    '''

    with open('testset1_scores.csv', 'w') as scorefile:
        scorefile.write("PSNR, " + str(thisPSNR) + '\n')
        scorefile.write("NIQE, " + str(thisNIQE) + '\n')
        scorefile.write("FID, " + str(thisFID) + '\n')
        scorefile.write("SSIM, " + str(thisSSIM) + '\n')
        scorefile.write("LPIPS, " + str(thisLPIPS) + '\n')

    print("results were written to scores.csv")
'''
