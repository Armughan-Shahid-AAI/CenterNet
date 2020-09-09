from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2

# DATA_PATH = '../../data/custom_kitti/'
DATA_PATH="/media/rameez/Drive1/armughan/incline/generate_3d_box_annotations/outputs/OD_1"
DATA_PATH="../../data/custom_coco"
SET_TO_SAME_CATEGORY = True

# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type             Describes the type of object: 'Car', 'Van/suv',
                         'Bus/truck', 'Trailer','Others'
   1    View Front/Rear  Integer Values from 0 to 6(inclusive) indication frontal view 
   1    View Side        Integer Values from 0 to 2(inclusive) indication side view 
   2    split_coords     coordinates for the split coordinate (x,y)
   4    bbox             2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
'''

'''
    Directory Structure
    
    custom_dataset
    --images
    
    --annotations
        --train.json
        --val.json
        --test.json
        --train
        --val
        --test
    --imgSets
        --train.txt
        --val.txt
        --test.txt
        
'''
def _bbox_to_coco_bbox(bbox):
    return [(bbox[0]), (bbox[1]),
            (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]




cats = [ 'Car', 'Van/suv', 'Bus/truck', 'Trailer','Others']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
if SET_TO_SAME_CATEGORY:
    cat_ids = {cat: 1 for i, cat in enumerate(cats)}


cat_info = []
if SET_TO_SAME_CATEGORY:
    cat_info.append({'name': "vehicle", 'id': 1, 'keypoints': ["split_point"]})
else:
    for i, cat in enumerate(cats):
            cat_info.append({'name': cat, 'id': i + 1, 'keypoints': ["split_point"]})


images_path = os.path.join(DATA_PATH , 'images')
ann_dir = os.path.join(DATA_PATH , 'annotations')
img_sets_path = os.path.join(DATA_PATH,"imgSets")
splits = ['train', 'val','test']

current_img_id = 0
for split in splits:
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    image_set = open(os.path.join(img_sets_path, '{}.txt'.format(split)), 'r')
    image_to_id = {}
    for line in image_set:
        if line[-1] == '\n':
            # line = line[:-1]
            line = line.strip().strip("\n")

        image_id = current_img_id
        image_info = {'file_name': line,
                      'id': int(image_id)}
        ret['images'].append(image_info)
        current_img_id+=1
        ann_path = os.path.join(ann_dir, split, '{}.txt'.format(os.path.splitext(line)[0]))
        # if split == 'val':
        #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
        anns = open(ann_path, 'r')


        for ann_ind, txt in enumerate(anns):
            # print (txt)
            # tmp = txt[:-1].split(' ')
            tmp = txt.strip("\n").strip().split(',')
            # print (tmp)
            # assert False
            cat_id = cat_ids[tmp[0]]
            view_front_rear = int(tmp[1])
            view_side = int(tmp[2])
            split_coords = [float(tmp[3]), float(tmp[4])]
            bbox = [float(tmp[5]), float(tmp[6]), float(tmp[7]), float(tmp[8])]

            ann = {'image_id': image_id,
                   'id': int(len(ret['annotations']) + 1),
                   'category_id': cat_id,
                   'bbox': _bbox_to_coco_bbox(bbox),
                   'split_coords': split_coords,
                   'view_front_rear': view_front_rear,
                   'view_side': view_side,
                   'keypoints': list(split_coords) + [2],
                   'num_keypoints': 1,

                   }
            ret['annotations'].append(ann)


    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    # import pdb; pdb.set_trace()
    out_path = os.path.join(ann_dir,'{}.json'.format(split))

    # out_path = '{}/annotations/custom_kitti_{}.json'.format(DATA_PATH, split)
    json.dump(ret, open(out_path, 'w'))


