import os
import sys
import argparse
import cv2
import glob
from tqdm import tqdm as tqdm

def create_dirs_if_not_exists(directories):
    if type(directories)==str:
        directories=[directories]

    for d in directories:
        if not os.path.isdir(d):
            os.makedirs(d)

def get_all_images_in_file(filePath):
   imgNames = None
   with open(filePath) as fp:
       imgNames=fp.readlines()
       imgNames = [i.strip().strip("\n") for i in imgNames]
   return imgNames

def main(cfg):
    input_img = cv2.imread(cfg.IMG_PATH)
    create_dirs_if_not_exists([cfg.OUTPUT_DIR])
    imgNames = get_all_images_in_file(cfg.TEXT_FILE)
    for imgName in tqdm(imgNames):
        basename = os.path.basename(imgName)
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, basename), input_img)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--TEXT_FILE', type=str,
                        help='Path to test video or images directory which need to be tested')
    parser.add_argument('--IMG_PATH', type=str,
                        help='Path to test video or images directory which need to be tested')

    parser.add_argument('--OUTPUT_DIR', type=str,
                        help='Type of input ')

    return parser.parse_args(argv)

if __name__ == '__main__':
    cfg = parse_arguments(sys.argv[1:])
    main(cfg)