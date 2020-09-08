from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.decode import det3d_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import det3d_post_process
from utils.debugger import Debugger
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

from .base_detector import BaseDetector


class Det3DDetector(BaseDetector):
    def __init__(self, opt):
        super(Det3DDetector, self).__init__(opt)


    def pre_process(self, image, scale, calib=None):
        height, width = image.shape[0:2]

        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([width / 2, height / 2], dtype=np.float32)
        if self.opt.keep_res:
            s = np.array([inp_width, inp_height], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = image  # cv2.resize(image, (width, height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = (inp_image.astype(np.float32) / 255.)
        inp_image = (inp_image - self.mean) / self.std
        images = inp_image.transpose(2, 0, 1)[np.newaxis, ...]

        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]

            print("xxxxxxxxxxxxxoutputxxxxxxxx\n")
            for n, t in output.items():

                print(n, " || ", t.size())
                print(n)
                print(t)
                print("\n\n")
            print("\nxxxxxxxxxxxxxxxxxxxxxxxx\n")
            output['hm'] = output['hm'].sigmoid_()
            output['sc'] = output['sc'].sigmoid_()
            output['vfr'] = output['vfr'].sigmoid_()
            output['vs'] = output['vs'].sigmoid_()

            wh = output['wh'] if self.opt.reg_bbox else None
            print("reg offset", self.opt.reg_offset)
            reg = output['reg'] if self.opt.reg_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()

            dets = det3d_decode(output['hm'], wh=wh, reg=reg, K=self.opt.K)
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        detections = det3d_post_process(
            dets.copy(), [meta['c']], [meta['s']], self.opt)
        return detections[0]

    def merge_outputs(self, detections):
        results = detections[0]
        # print ("merge outputs called line 86 detectoors.det3D", results)
        for j in range(1, self.num_classes + 1):
            if len(results[j] > 0):
                keep_inds = (results[j][:, -1] > self.opt.peak_thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy()
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img * self.std + self.mean) * 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        debugger.add_ct_detection(
            img, dets[0], show_box=self.opt.reg_bbox,
            center_thresh=self.opt.vis_thresh, img_id='det_pred')

    def show_results(self, debugger, image, results):
        debugger.add_3d_detection(
            image, results, self.this_calib,
            center_thresh=self.opt.vis_thresh, img_id='add_pred')

        debugger.show_all_imgs(pause=self.pause)