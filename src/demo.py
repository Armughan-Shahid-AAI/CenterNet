from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory
import time

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def create_dirs_if_not_exists(directories):
    if type(directories)==str:
        directories=[directories]

    for d in directories:
        if not os.path.isdir(d):
            os.makedirs(d)

def deleteFolders(folders,checkExists=False):
    if checkExists:
        for folder in folders:
            if os.path.exists(folder):
                os.system("rm -r '{}' ".format(os.path.normpath(folder)))
    else:

        for folder in folders:
            os.system("rm -r '{}' ".format(os.path.normpath(folder)))

def emptyFolders(folders,checkExists=True):
    deleteFolders(folders,checkExists=checkExists)
    for folder in folders:
        os.makedirs(folder)

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  output_dir = "../results"
  create_dirs_if_not_exists([output_dir])
  emptyFolders([output_dir])

  calib = np.array([7.215377000000e+02,
  0.000000000000e+00,
  6.095593000000e+02,
  4.485728000000e+01,
  0.000000000000e+00,
  7.215377000000e+02,
  1.728540000000e+02,
  2.163791000000e-01,
  0.000000000000e+00,
  0.000000000000e+00,
  1.000000000000e+00,
  2.745884000000e-03]).reshape(3,4).astype(np.float32)

  calib = np.array([529.5349731445312, 0.0, 619.1300048828125,
                    0.0, 0.0, 529.260009765625, 367.114990234375,
                    0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3,4).astype(np.float32)
  print ("calib  ", calib)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name, meta=calib, output_img_path = os.path.join(output_dir,os.path.basename(image_name)))
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
