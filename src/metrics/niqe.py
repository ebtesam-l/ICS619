from basicsr.metrics.niqe import calculate_niqe
import cv2
import glob 
import numpy as np 
from tqdm import tqdm

import argparse



def calculate_images_niqe(dir): 
    """This method calculate NIQE for a list of images within a directory. 
        Returns the mean of all images NIQE

    Args:
        dir (str): dir

    Returns:
        float: the mean of all Images NIQE. 
    """    
    dir_images = f'{dir}/*'
    images = glob.glob(dir_images)

    niqe_list = list()
    for image in images: #tqdm(images,desc='Calculate NIQE '):
        img = cv2.imread(image)
        niqe_list.append(calculate_niqe(img,0, verbose=False))
        # print(calculate_niqe(img,0))
    return np.mean(niqe_list)

if __name__ == "__main__": 
    argParser = argparse.ArgumentParser()
    argParser.add_argument( "--dirEI", help="directory containing enhanced images, example: enhanced/", required = True)
    
    args = argParser.parse_args()

    dir = args.dirEI
    print(calculate_images_niqe(dir))
