# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from numpy.lib.twodim_base import tri
import torch
from mmcv.runner import load_checkpoint

from .. import build_detector
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector
from mmrotate.core import rbbox2result
import torch.nn as nn
from mmdet.utils import get_root_logger
from mmcv.cnn import constant_init, kaiming_init
import torch.nn.functional as F
from mmcv.ops import DeformConv2d

@ROTATED_DETECTORS.register_module()
class KDAttAngle_RotatedSingleStageDetector(RotatedSingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.
    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,

                 teacher_config,
                 teacher_ckpt,

                 # teacher_model,
                 # tea_pretrained=None,

                 # idea s_att*L2[s,t]
                 ofc_weight=0.1,
                 temp_stu=0.1,
                 temp_tea=0.1,

                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(KDAttAngle_RotatedSingleStageDetector, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)

        # Build teacher model
        self.eval_teacher = True

        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')

        # self.teacher_model = build_detector(teacher_model, train_cfg=train_cfg, test_cfg=test_cfg)
        # self.load_weights(pretrained=tea_pretrained)
        self.freeze_models()

        # idea
        self.ofc_weight = ofc_weight
        self.temp_stu = temp_stu
        self.temp_tea = temp_tea
        self.att_adaptive_layers = nn.ModuleList([Trans(256, 256, kernel_size=3, stride=1, padding=1),
                                                  Trans(256, 256, kernel_size=3, stride=1, padding=1),
                                                  Trans(256, 256, kernel_size=3, stride=1, padding=1),
                                                  Trans(256, 256, kernel_size=3, stride=1, padding=1),
                                                  Trans(256, 256, kernel_size=3, stride=1, padding=1)])

        self.init_adap_weights()


    def init_adap_weights(self, mean=0, std=0.0001):
        if self.ofc_weight is not None:
            for m in self.att_adaptive_layers:
                # m.weight.data.normal_().fmod_(2).mul_(std).add_(mean)
                m.conv2.weight.data.normal_().fmod_(2).mul_(std).add_(mean)
                m.conv1.weight.data.normal_().fmod_(2).mul_(std).add_(mean)

    def load_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self.teacher_model, pretrained, strict=False, logger=logger)
            for name, parameters in self.teacher_model.named_parameters():
                if name == "neck.lateral_convs.1.conv.bias":
                    print(parameters)
            print("load teacher model success")

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        #super(KnowledgeDistillationRotatedSingleStageDetector, self).forward_train(img, img_metas)
        self.teacher_model.eval()
        with torch.no_grad():
            tea_x = self.teacher_model.extract_feat(img)
            # print("tea_x", tea_x[0].size())
            tea_outs = self.teacher_model.bbox_head(tea_x)

        stu_x = self.extract_feat(img)
        # stu_outs = self.bbox_head(stu_x)
        # print("stu_x", stu_x[0].size())

        losses = self.bbox_head.forward_train(stu_x, tea_outs, tea_x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)


        att_loss = 0.0

        for i in range(len(stu_x)):
            '''idea att_sptial_mask'''
            if self.ofc_weight is not None:
                stu_spatial_att = self.mask_att_spatial(stu_x[i], s_t=self.temp_stu)
                tea_spatial_att = self.mask_att_spatial(tea_x[i], s_t=self.temp_tea)
                # spatial
                sum_spatial_att = (stu_spatial_att + tea_spatial_att) / 2
                sum_spatial_att = sum_spatial_att.detach()

                tmp = (self.att_adaptive_layers[i](stu_x[i]) - tea_x[i]) ** 2 * sum_spatial_att
                att_loss += (torch.sum(tmp) ** 0.5) * self.ofc_weight

        if self.ofc_weight is not None:
            losses.update(dict(loss_att=att_loss))

        return losses

    def mask_att_spatial(self, x, s_t):  # 输入是是一个fpn层的分类输出(N,9*num_class,H,W)
        # s_t是温度系数
        spatial_att = torch.mean(torch.abs(x), dim=1, keepdim=True)  # (N,1,H,W)
        s_size = spatial_att.size()
        spatial_att = spatial_att.view(x.size(0), -1)

        spatial_att = torch.softmax(spatial_att / s_t, dim=1) * s_size[2] * s_size[3]
        spatial_att = spatial_att.view(s_size)
        return spatial_att


    def freeze_models(self):
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

class Trans(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(Trans, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out