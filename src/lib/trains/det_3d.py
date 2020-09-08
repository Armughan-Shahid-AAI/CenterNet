from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, L1Loss, BinRotLoss, CrossEntropyLossWMask
from models.decode import ddd_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ddd_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class Det3dLoss(torch.nn.Module):
    def __init__(self, opt):
        super(Det3dLoss, self).__init__()
        print ("Using MSE loss for clasification = ",opt.mse_loss)
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_view_side = CrossEntropyLossWMask()
        self.crit_view_front_rear = CrossEntropyLossWMask()
        self.crit_reg = L1Loss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        # for i,v in batch.items():
        #     if i in ['hm', 'vfr','vs','sc']:
        #         print (i, v.size())
        hm_loss, sc_loss, vs_loss, vfr_loss = 0, 0, 0, 0
        # vs_loss=torch.tensor(vs_loss,dtype=torch.float)
        # vfr_loss=torch.tensor(vfr_loss,dtype=torch.float)
        wh_loss, off_loss = 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            output['sc'] = output['sc'].sigmoid_()

            # if opt.eval_oracle_dep:
            #     output['dep'] = torch.from_numpy(gen_oracle_map(
            #         batch['dep'].detach().cpu().numpy(),
            #         batch['ind'].detach().cpu().numpy(),
            #         opt.output_w, opt.output_h)).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.sc_weight>0:
                sc_loss += self.crit_reg(output['sc'], batch['sc_mask'],
                                          batch['ind'], batch['sc']) / opt.num_stacks

            if opt.vfr_weight > 0:
                vfr_loss += self.crit_view_front_rear(output['vfr'], batch['reg_mask'],
                                         batch['ind'], batch['vfr'], 7)
                # vfr_loss += self.crit_view_front_rear(output['vfr'], batch['vfr']) / opt.num_stacks

            if opt.vs_weight > 0:
                vs_loss += self.crit_view_side(output['vs'], batch['reg_mask'],
                                                      batch['ind'], batch['vs'], 3)
                # vs_loss += self.crit_view_side(output['vs'], batch['vs']) / opt.num_stacks


            if opt.reg_bbox and opt.wh_weight > 0:
                wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                         batch['ind'], batch['wh']) / opt.num_stacks
            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.sc_weight * sc_loss + \
               opt.vfr_weight * vfr_loss + opt.vs_weight * vs_loss + \
               opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'sc_loss': sc_loss,
                      'vfr_loss': vfr_loss, 'vs_loss': vs_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}

        # print ("\n\n\nlosss")
        # print (loss_stats)
        # print ("\nloss ended\n\n\n")
        return loss, loss_stats


class Det_3d(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(Det_3d, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'sc_loss', 'vfr_loss', 'vs_loss',
                       'wh_loss', 'off_loss']
        loss = Det3dLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        print ("callled det 3d debugger in trains/det_3d")
        # opt = self.opt
        # wh = output['wh'] if opt.reg_bbox else None
        # reg = output['reg'] if opt.reg_offset else None
        # dets = ddd_decode(output['hm'], output['rot'], output['dep'],
        #                   output['dim'], wh=wh, reg=reg, K=opt.K)
        #
        # # x, y, score, r1-r8, depth, dim1-dim3, cls
        # dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        # calib = batch['meta']['calib'].detach().numpy()
        # # x, y, score, rot, depth, dim1, dim2, dim3
        # # if opt.dataset == 'gta':
        # #   dets[:, 12:15] /= 3
        # dets_pred = ddd_post_process(
        #     dets.copy(), batch['meta']['c'].detach().numpy(),
        #     batch['meta']['s'].detach().numpy(), calib, opt)
        # dets_gt = ddd_post_process(
        #     batch['meta']['gt_det'].detach().numpy().copy(),
        #     batch['meta']['c'].detach().numpy(),
        #     batch['meta']['s'].detach().numpy(), calib, opt)
        # # for i in range(input.size(0)):
        # for i in range(1):
        #     debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug == 3),
        #                         theme=opt.debugger_theme)
        #     img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
        #     img = ((img * self.opt.std + self.opt.mean) * 255.).astype(np.uint8)
        #     pred = debugger.gen_colormap(
        #         output['hm'][i].detach().cpu().numpy())
        #     gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
        #     debugger.add_blend_img(img, pred, 'hm_pred')
        #     debugger.add_blend_img(img, gt, 'hm_gt')
        #     # decode
        #     debugger.add_ct_detection(
        #         img, dets[i], show_box=opt.reg_bbox, center_thresh=opt.center_thresh,
        #         img_id='det_pred')
        #     debugger.add_ct_detection(
        #         img, batch['meta']['gt_det'][i].cpu().numpy().copy(),
        #         show_box=opt.reg_bbox, img_id='det_gt')
        #     debugger.add_3d_detection(
        #         batch['meta']['image_path'][i], dets_pred[i], calib[i],
        #         center_thresh=opt.center_thresh, img_id='add_pred')
        #     debugger.add_3d_detection(
        #         batch['meta']['image_path'][i], dets_gt[i], calib[i],
        #         center_thresh=opt.center_thresh, img_id='add_gt')
        #     # debugger.add_bird_view(
        #     #   dets_pred[i], center_thresh=opt.center_thresh, img_id='bird_pred')
        #     # debugger.add_bird_view(dets_gt[i], img_id='bird_gt')
        #     debugger.add_bird_views(
        #         dets_pred[i], dets_gt[i],
        #         center_thresh=opt.center_thresh, img_id='bird_pred_gt')
        #
        #     # debugger.add_blend_img(img, pred, 'out', white=True)
        #     debugger.compose_vis_add(
        #         batch['meta']['image_path'][i], dets_pred[i], calib[i],
        #         opt.center_thresh, pred, 'bird_pred_gt', img_id='out')
        #     # debugger.add_img(img, img_id='out')
        #     if opt.debug == 4:
        #         debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
        #     else:
        #         debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        print("train/det_3d save fn called")
        # opt = self.opt
        # wh = output['wh'] if opt.reg_bbox else None
        # reg = output['reg'] if opt.reg_offset else None
        # dets = ddd_decode(output['hm'], output['rot'], output['dep'],
        #                   output['dim'], wh=wh, reg=reg, K=opt.K)
        #
        # # x, y, score, r1-r8, depth, dim1-dim3, cls
        # dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        # calib = batch['meta']['calib'].detach().numpy()
        # # x, y, score, rot, depth, dim1, dim2, dim3
        # dets_pred = ddd_post_process(
        #     dets.copy(), batch['meta']['c'].detach().numpy(),
        #     batch['meta']['s'].detach().numpy(), calib, opt)
        # img_id = batch['meta']['img_id'].detach().numpy()[0]
        # results[img_id] = dets_pred[0]
        # for j in range(1, opt.num_classes + 1):
        #     keep_inds = (results[img_id][j][:, -1] > opt.center_thresh)
        #     results[img_id][j] = results[img_id][j][keep_inds]