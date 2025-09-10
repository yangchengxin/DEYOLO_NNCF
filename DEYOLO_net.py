# -*- coding: UTF-8 -*-
"""
  @Author: ycx
  @Date  : 2025/9/10 16:43
  @Email : <ycx97124@163.com>
  @version V1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus,
                                    GhostBottleneck, GhostConv, HGBlock, HGStem, RepC3, RepConv, DEA, C2f_BiFocus)
from ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.yolo.utils.loss import v8DetectionLoss
from ultralytics.yolo.utils.plotting import feature_visualization
from ultralytics.yolo.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                                intersect_dicts, make_divisible, model_info, scale_img, time_sync)

class DEYOLO(nn.Module):
    def __init__(self, nc:int = 80, depth_ratio:float = 0.33, width_ratio:float = 0.25, reg_max:int = 16):
        super(DEYOLO, self).__init__()
        self.save = []

        # ---- backbone1-vi ---- #                                 feature map idx/stride
        self.cv1    = Conv(c1 = 3, c2 = 16, k = 3, s = 2)                 # 0 p1/2            1 * 3 * 640 * 640  -> 1 * 16 * 320 * 320
        self.cv2    = Conv(c1 = 16, c2 = 32, k = 3, s = 2)                # 1 p2/4            1 * 16 * 320 * 320 -> 1 * 32 * 160 * 160
        self.c2fb_1 = C2f_BiFocus(c1 = 32, c2 = 32, n = True)             # 2                 1 * 32 * 160 * 160 -> 1 * 32 * 160 * 160
        self.cv3    = Conv(c1 = 32, c2 = 64, k = 3, s = 2)                # 3 p3/8            1 * 32 * 160 * 160 -> 1 * 64 * 80 * 80
        self.c2f_1  = C2f(c1 = 64, c2 = 64, n = 2, shortcut = True)       # 4                 1 * 64 * 80 * 80   -> 1 * 64 * 80 * 80
        self.cv4    = Conv(c1 = 64, c2 = 128, k = 3, s = 2)               # 5 p4/16           1 * 64 * 80 * 80   -> 1 * 128 * 40 * 40
        self.c2f_2  = C2f(c1 = 128, c2 = 128, n = 2, shortcut = True)     # 6                 1 * 128 * 40 * 40  -> 1 * 128 * 40 * 40
        self.cv5    = Conv(c1 = 128, c2 = 256, k = 3, s = 2)              # 7 p5/32           1 * 128 * 40 * 40  -> 1 * 256 * 20 * 20
        self.c2f_3  = C2f(c1 = 256, c2 = 256, n = 1, shortcut = True)     # 8                 1 * 256 * 20 * 20  -> 1 * 256 * 20 * 20
        self.sppf_1 = SPPF(c1 = 256, c2 = 256)                            # 9                 1 * 256 * 20 * 20  -> 1 * 256 * 20 * 20

        # ---- backbone2-ir ---- #
        self.cv6    = Conv(c1 = 3, c2 = 16, k = 3, s = 2)                 # 10 p1/2            1 * 3 * 640 * 640  -> 1 * 16 * 320 * 320
        self.cv7    = Conv(c1 = 16, c2 = 32, k = 3, s = 2)                # 11 p2/4            1 * 16 * 320 * 320 -> 1 * 32 * 160 * 160
        self.c2fb_2 = C2f_BiFocus(c1 = 32, c2 = 32, n = True)             # 12                 1 * 32 * 160 * 160 -> 1 * 32 * 160 * 160
        self.cv8    = Conv(c1 = 32, c2 = 64, k = 3, s = 2)                # 13 p3/8            1 * 32 * 160 * 160 -> 1 * 64 * 80 * 80
        self.c2f_4  = C2f(c1 = 64, c2 = 64, n = 2, shortcut = True)       # 14                 1 * 64 * 80 * 80   -> 1 * 64 * 80 * 80
        self.cv9    = Conv(c1 = 64, c2 = 128, k = 3, s = 2)               # 15 p4/16           1 * 64 * 80 * 80   -> 1 * 128 * 40 * 40
        self.c2f_5  = C2f(c1 = 128, c2 = 128, n = 2, shortcut = True)     # 16                 1 * 128 * 40 * 40  -> 1 * 128 * 40 * 40
        self.cv10   = Conv(c1 = 128, c2 = 256, k = 3, s = 2)              # 17 p5/32           1 * 128 * 40 * 40  -> 1 * 256 * 20 * 20
        self.c2f_6  = C2f(c1 = 256, c2 = 256, n = 1, shortcut = True)     # 18                 1 * 256 * 20 * 20  -> 1 * 256 * 20 * 20
        self.sppf_2 = SPPF(c1 = 256, c2 = 256)                            # 19                 1 * 256 * 20 * 20  -> 1 * 256 * 20 * 20

        # ---- neck ---- #
        # attention
        self.dea_1  = DEA(channel = 64, kernel_size = 80)                 # 20
        self.dea_2  = DEA(channel = 128, kernel_size = 40)                # 21
        self.dea_3  = DEA(channel = 256, kernel_size = 20)                # 22
        # attention

        self.ups_1  = nn.Upsample(scale_factor = 2, mode = 'nearest')     # 23
        self.cat_1  = Concat(dimension=1)                                 # 24
        self.c2f_7  = C2f(c1 = 384, c2 = 128, n = 1, shortcut = False)    # 25

        self.ups_2  = nn.Upsample(scale_factor = 2, mode = 'nearest')     # 26
        self.cat_2  = Concat(dimension=1)                                 # 27
        self.c2f_8  = C2f(c1 = 192, c2 = 64, n = 1, shortcut = False)     # 28 p3/8 small

        self.cv11  = Conv(c1 = 64, c2 = 64, k = 3, s = 2)                 # 29
        self.cat_3 = Concat(dimension=1)                                  # 30
        self.c2f_9 = C2f(c1 = 192, c2 = 128, n = 1, shortcut = False)     # 31 p4/16 medium

        self.cv12  = Conv(c1 = 128, c2 = 128, k = 3, s = 2)               # 32
        self.cat_4 = Concat(dimension=1)                                  # 33
        self.c2f_10 = C2f(c1 = 384, c2 = 256, n = 1, shortcut = False)    # 34 p5/32 large


        # ---- head ---- #
        self.head  = Detect(nc = nc, ch =[64, 128, 256])                  # 35 detect head (p3 p4 p5)

    def forward(self, x1, x2):
        # ---- vis feature ---- #
        out0 = self.cv1(x1)
        out1 = self.cv2(out0)
        out2 = self.c2fb_1(out1)
        out3 = self.cv3(out2)
        out4 = self.c2f_1(out3)
        out5 = self.cv4(out4)
        out6 = self.c2f_2(out5)
        out7 = self.cv5(out6)
        out8 = self.c2f_3(out7)
        out9 = self.sppf_1(out8)

        # ---- ir feature ---- #
        out10 = self.cv6(x2)
        out11 = self.cv7(out10)
        out12 = self.c2fb_2(out11)
        out13 = self.cv8(out12)
        out14 = self.c2f_4(out13)
        out15 = self.cv9(out14)
        out16 = self.c2f_5(out15)
        out17 = self.cv10(out16)
        out18 = self.c2f_6(out17)
        out19 = self.sppf_2(out18)

        # DEA
        out20 = self.dea_1([out4, out14])
        out21 = self.dea_2([out6, out16])
        out22 = self.dea_3([out9, out19])

        # FPN + PAN
        out23 = self.ups_1(out22)
        out24 = self.cat_1([out23, out21])
        out25 = self.c2f_7(out24)

        out26 = self.ups_2(out25)
        out27 = self.cat_2([out26, out20])
        out28 = self.c2f_8(out27)

        out29 = self.cv11(out28)
        out30 = self.cat_3([out29, out25])
        out31 = self.c2f_9(out30)

        out32 = self.cv12(out31)
        out33 = self.cat_4([out32, out22])
        out34 = self.c2f_10(out33)

        # head
        out35 = self.head([out28, out31, out34])

        return out35





