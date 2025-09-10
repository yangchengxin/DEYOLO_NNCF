# -*- coding: UTF-8 -*-
"""
  @Author: ycx
  @Date  : 2025/9/9 14:53
  @Email : <ycx97124@163.com>
  @version V1.0
"""

from email.policy import strict

import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic
import onnx
import numpy as np
import nncf
from ultralytics import YOLO
from PTQ_NNCF_implicit import create_data_source, transform_fn
from DEYOLO_net import DEYOLO

# -------------------------------------------------
# 1. 滑动平均校准器
# -------------------------------------------------
class MovingAverageMinMax:
    def __init__(self, momentum=0.9):
        self.min_val = None
        self.max_val = None
        self.momentum = momentum

    def update(self, x):
        with torch.no_grad():
            cur_min, cur_max = x.min(), x.max()
            if self.min_val is None:
                self.min_val, self.max_val = cur_min, cur_max
            else:
                self.min_val = self.momentum * self.min_val + (1 - self.momentum) * cur_min
                self.max_val = self.momentum * self.max_val + (1 - self.momentum) * cur_max

    def compute_qparams(self, dtype=torch.qint8):
        scale = (self.max_val - self.min_val) / 127
        zero_point = torch.round(-self.min_val / scale).to(torch.int8)
        return scale, zero_point

# -------------------------------------------------
# 2. 伪量化模块（Quantize + DeQuantize）
# -------------------------------------------------
class FakeQuantize(nn.Module):
    def __init__(self, observer):
        super().__init__()
        self.observer = observer
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0, dtype=torch.int8))

    def forward(self, x):
        if self.training:          # 收集阶段
            self.observer.update(x)
        scale, zp = self.observer.compute_qparams()
        self.scale.copy_(scale)
        self.zero_point.copy_(zp)
        # 伪量化：先量化再反量化
        x_q = torch.quantize_per_tensor(x, scale, zp, torch.qint8)
        return x_q.dequantize()

# -------------------------------------------------
# 3. 把校准器插到模型里
# -------------------------------------------------
def add_fake_quant(model, target_module=nn.Conv2d):
    """
    在每一层 Conv 后插入 FakeQuantize
    """
    for name, m in model.named_children():
        if isinstance(m, target_module):
            # 为权重也建一个校准器
            w_obs = MovingAverageMinMax()
            fq_w = FakeQuantize(w_obs)
            # 把权重包起来
            m.weight = nn.Parameter(fq_w(m.weight))
            # 为激活再建一个
            a_obs = MovingAverageMinMax()
            fq_a = FakeQuantize(a_obs)
            # 插入到 forward 路径
            m.register_forward_hook(lambda module, inp, out: fq_a(out))
        else:
            add_fake_quant(m, target_module)


if __name__ == '__main__':

    # -------------------------------------------------
    # 4. 生成校准数据集对象
    # -------------------------------------------------
    data_source = create_data_source()

    # -------------------------------------------------
    # 5. 载入 FP32 模型并跑校准
    # -------------------------------------------------
    model = DEYOLO(nc=80)

    checkpoint = torch.load(r'D:\ycx_git_repositories\DEYOLO_NNCF\DEYOLO\runs\detect\train\weights\best.pt', map_location='cpu')
    # 提取模型的 state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model'].state_dict()
    else:
        state_dict = checkpoint.state_dict()

    model.load_state_dict(state_dict, strict=False)   # 你的 FP32 权重
    model.eval()
    add_fake_quant(model)

    # 跑 100 张图即可完成校准
    for i, img in enumerate(data_source):   # img: tensor [N,3,H,W]
        with torch.no_grad():
            _ = model(img['img'].to(torch.float32), img['img2'].to(torch.float32))
        if i >= 99:
            break

    # -------------------------------------------------
    # 6. 导出带 Q/DQ 的 ONNX（显式量化）
    # -------------------------------------------------
    vi_input = torch.randn(1, 3, 640, 640)
    ir_input = torch.randn(1, 3, 640, 640)
    dummy_input = (vi_input, ir_input)
    torch.onnx.export(model, dummy_input, 'best_int8_explicit.onnx',
                      opset_version=11,
                      do_constant_folding=False,   # 保留 Q/DQ 节点
                      input_names=['images1', 'images2'],
                      output_names=['output0'])