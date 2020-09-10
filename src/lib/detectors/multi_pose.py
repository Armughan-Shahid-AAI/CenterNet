from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class MultiPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector, self).__init__(opt)
    self.flip_idx = opt.flip_idx
    self.num_joints = opt.num_joints

    vfr_to_int_mapper = {
      "None": 0,
      "Front - Left BB part": 1,
      "Rear - Left BB part": 2,
      "Front - Right BB part": 3,
      "Rear - Right BB part": 4,
      "Front Only": 5,
      "Rear Only": 6,

    }
    vs_to_int_mapper = {
      "None": 0,
      "Left": 1,
      "Right": 2,

    }
    self.int_to_vfr_mapper = {v: k for k, v in vfr_to_int_mapper.items()}
    self.int_to_vs_mapper = {v: k for k, v in vs_to_int_mapper.items()}


  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      if self.opt.hm_hp and not self.opt.mse_loss:
        output['hm_hp'] = output['hm_hp'].sigmoid_()

      reg = output['reg'] if self.opt.reg_offset else None
      hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      if self.opt.flip_test:
        output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
        hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                if hm_hp is not None else None
        reg = reg[0:1] if reg is not None else None
        hp_offset = hp_offset[0:1] if hp_offset is not None else None
      
      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'], output['vfr'], output['vs'],
        reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'], num_joints=self.num_joints)
    for j in range(1, self.num_classes + 1):
      ##changed
      # dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
      ##changed
      # dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5+(self.num_joints*2))
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5+(self.num_joints*2)+2)
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    if self.opt.nms or len(self.opt.test_scales) > 1:
      soft_nms_39(results[1], Nt=0.5, method=2)
    results[1] = results[1].tolist()
    print ("merger shape" ,np.array(results[1]).shape)
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:39] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')

  def get_3d_info(self,bbox, view_front_rear, view_side, split_coord):
    view_front_mapper, view_side_mapper = self.int_to_vfr_mapper, self.int_to_vs_mapper
    min_x, min_y, max_x, max_y = bbox
    w, h = max_x - min_x, max_y - min_y
    view_front_rear = view_front_mapper[view_front_rear]
    view_side = view_side_mapper[view_side]

    rotation_ry = 0
    delta_x = 0
    delta_y = 0
    delta_z = 0
    sigma = 1e-6
    center_x_3d = min_x + (w / 2)
    center_y_3d = (min_y + (h / 2))

    split_coord_x, split_coord_y = split_coord
    split_coord_delta_x = split_coord_x - min_x
    split_coord_delta_y = split_coord_y - min_y

    if view_front_rear == "Front - Left BB part":
      delta_x = split_coord_delta_x
      delta_z = (((w - split_coord_delta_x) ** 2) + ((h - split_coord_delta_y) ** 2)) ** 0.5
      delta_y = split_coord_delta_y
      rotation_ry = -(np.pi - np.arcsin((h - split_coord_delta_y) / (delta_z + sigma)))


    elif view_front_rear == "Rear - Left BB part":
      delta_x = split_coord_delta_x
      delta_z = (((w - split_coord_delta_x) ** 2) + ((h - split_coord_delta_y) ** 2)) ** 0.5
      delta_y = split_coord_delta_y
      rotation_ry = np.arcsin((h - split_coord_delta_y) / (delta_z + sigma))

    elif view_front_rear == "Front - Right BB part":
      delta_x = w - split_coord_delta_x
      delta_z = (((split_coord_delta_x) ** 2) + ((h - split_coord_delta_y) ** 2)) ** 0.5
      delta_y = split_coord_delta_y
      rotation_ry = -np.arcsin((h - split_coord_delta_y) / (delta_z + sigma))

    elif view_front_rear == "Rear - Right BB part":
      delta_x = w - split_coord_delta_x
      delta_z = (((split_coord_delta_x) ** 2) + ((h - split_coord_delta_y) ** 2)) ** 0.5
      delta_y = split_coord_delta_y
      rotation_ry = np.pi - np.arcsin((h - split_coord_delta_y) / (delta_z + sigma))

    elif view_front_rear == "Front Only":
      delta_x = w
      delta_z = 0
      delta_y = h
      rotation_ry = -(np.pi / 2)

    elif view_front_rear == "Rear Only":
      delta_x = w
      delta_z = 0
      delta_y = h
      rotation_ry = (np.pi / 2)

    elif view_front_rear == "None":
      delta_x = 0
      delta_z = w
      delta_y = h
      rotation_ry = np.pi if view_side == "Left" else 0

    return min_x, min_y, w, h, center_x_3d, center_y_3d, delta_x, delta_y, delta_z, rotation_ry

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='multi_pose')
    for bbox in results[1]:
      if bbox[4] > self.opt.vis_thresh:
        vfr = bbox[-2]
        vs = bbox[-1]
        split_coord = bbox[5:5+(2*self.num_joints)]
        # debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose', text=":{}, {}".format(
        #   self.int_to_vfr_mapper[vfr], self.int_to_vs_mapper[vs])
        #                        , show_classname=False)

        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose', text=":{}".format(
          self.int_to_vs_mapper[vs])
                               , show_classname=False)

        ##changed
        # debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
        debugger.add_coco_hp(bbox[5:5+(2*self.num_joints)], img_id='multi_pose')
        # info = self.get_3d_info(bbox[:4], vfr, vs, split_coord)
        # debugger.add_2d_to_3d_detection(info, img_id='multi_pose' )
        debugger.add_orientation_lines(split_coord, bbox[:4], img_id='multi_pose')
    debugger.show_all_imgs(pause=self.pause)