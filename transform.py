import os
import sys
import argparse
import cv2
import numpy as np
import glob
from tqdm import tqdm as tqdm

def create_dirs_if_not_exists(directories):
    if type(directories)==str:
        directories=[directories]

    for d in directories:
        if not os.path.isdir(d):
            os.makedirs(d)

def get_all_imgs_in_tree(folderPath, extensions = ['jpg','png'], getCompletePaths=True):
    img_names = []
    for ext in extensions:
        img_names += glob.glob("{}/**/*.{}".format(folderPath, ext), recursive=True)
    if not getCompletePaths:
        img_names = [os.path.basename(i) for i in img_names]
    return img_names

def main(cfg):
    inputDir = cfg.INPUT_DIR
    imgNames = get_all_imgs_in_tree(inputDir, getCompletePaths=True)
    create_dirs_if_not_exists([cfg.OUTPUT_DIR])
    for imgName in tqdm(imgNames):
        img=cv2.imread(imgName)
        img = cv2.resize(img, (cfg.R_WIDTH, cfg.R_HEIGHT))
        basename = os.path.basename(imgName)
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, basename), img)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--INPUT_DIR', type=str,
                        help='Path to test video or images directory which need to be tested')

    parser.add_argument('--OUTPUT_DIR', type=str,
                        help='Type of input ')

    parser.add_argument('--R_WIDTH', type=int,
                        help='width of transformed img', default= 1280)

    parser.add_argument('--R_HEIGHT', type=int,
                        help='Height of transformed img', default=720)

    return parser.parse_args(argv)

if __name__ == '__main__':
    cfg = parse_arguments(sys.argv[1:])
    main(cfg)