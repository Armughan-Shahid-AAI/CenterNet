from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
cv2.setNumThreads(0)
import os
import math
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import pycocotools.coco as coco


class Det3dDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def __getitem__(self, index):
        # print ("get item called for index ", index)
        img_id = self.images[index]
        # print("img id ",img_id)
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        # print ("img info ", img_info)
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        # print ("img path ", img_path)
        assert os.path.exists(img_path),"img path {} doesnot exist".format(img_path)
        # print ("reading image")
        img = cv2.imread(img_path)
        # print ("img read. image shape ",img.shape)
        # img=cv2.resize(img, (1280,380))
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
        if self.opt.keep_res:
            s = np.array([self.opt.input_w, self.opt.input_h], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)

        aug = False
        if self.split == 'train' and np.random.random() < self.opt.aug_ddd:
            aug = True
            sf = self.opt.scale
            cf = self.opt.shift
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            c[0] += img.shape[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += img.shape[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        # print("beginnning annotation generation")

        # print ("getting affine transforamtion for input")
        trans_input = get_affine_transform(
            c, s, 0, [self.opt.input_w, self.opt.input_h])
        # print ("generated affine transformation for input")
        # print ("warping image", trans_input)
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_w, self.opt.input_h),
                             flags=cv2.INTER_LINEAR)
        # print ("image warped")

        inp = (inp.astype(np.float32) / 255.)
        # if self.split == 'train' and not self.opt.no_color_aug:
        #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        # print ("image transposed")

        num_classes = self.opt.num_classes
        # print ("getting affine transforamtion for output")

        trans_output = get_affine_transform(
            c, s, 0, [self.opt.output_w, self.opt.output_h])
        # print("transformations generated")

        hm = np.zeros(
            (num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
        vs = np.zeros((3,self.opt.output_h, self.opt.output_w), dtype=np.float32)
        vfr = np.zeros((7,self.opt.output_h, self.opt.output_w), dtype=np.float32)

        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        sc = np.zeros((self.max_objs, 2), dtype=np.float32)

        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        sc_mask = np.zeros((self.max_objs), dtype=np.uint8)
        # print ("getting annotation id")
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        # print ("retreived annotation ids")
        # print ("loading annotation for image")
        anns = self.coco.loadAnns(ids=ann_ids)
        # print ("annotations loaded for image")

        num_objs = min(len(anns), self.max_objs)
        # print ("num of objects ", num_objs)
        # print ("drawing gaussian")
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian
        # print ("gaussian drawn")
        gt_det = []
        for k in range(num_objs):
            # print ("Object number ", k)
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            split_coordinates = ann['split_coords']
            # print ("initial split coordinates ", split_coordinates)
            view_front_rear = ann['view_front_rear']
            view_side = ann['view_side']

            # if flipped:
            #   bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            # print ("affine transforming object")
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((h, w))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                draw_gaussian(hm[cls_id], ct, radius)
                draw_gaussian(vfr[view_front_rear], ct, radius)
                draw_gaussian(vs[view_side], ct, radius)

                # vfr[k][view_front_rear] = view_front_rear
                # vs[k] = view_side

                wh[k] = 1. * w, 1. * h
                gt_det.append([ct[0], ct[1]] + [cls_id])
                # print ("regresssing width height status",self.opt.reg_bbox)
                if self.opt.reg_bbox:
                    gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]

                ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1 #if not aug else 0

                # print("split_coordinates input ",split_coordinates)
                if view_front_rear<4:
                    split_coordinates = affine_transform(split_coordinates, trans_output)
                    # print("split coordinates transformed ", split_coordinates)
                    # print ("output split coordinates ", split_coordinates)
                    split_coordinates = (split_coordinates-bbox[:2])/wh[k]
                    # print("split coordinates deltas ", split_coordinates)
                    sc_mask[k]=1

                sc[k] = split_coordinates

        # print ("objs gt generated")
        # print('gt_det', gt_det)
        # print("input size ########",inp.shape)
        ret = {'input': inp, 'hm': hm, 'ind': ind,
                'reg_mask': reg_mask, "sc": sc,
               "sc_mask": sc_mask, 'vfr': vfr,
               'vs': vs
               }

        # print ("reg bbox w h = ",self.opt.reg_bbox)
        if self.opt.reg_bbox:
            ret.update({'wh': wh})
        # print("reg bbox center offset = ", self.opt.reg_offset)
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not ('train' in self.split):
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 5), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det,
                    'image_path': img_path, 'img_id': img_id}
            ret['meta'] = meta
        # print("annotation generated")
        return ret

