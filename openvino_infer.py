# -*- coding: UTF-8 -*-
"""
  @Author: ycx
  @Date  : 2025/9/9 14:25
  @Email : <ycx97124@163.com>
  @version V1.0
"""

import cv2
import time
import yaml
import torch
import argparse
import requests
import numpy as np
import openvino as ov
from pathlib import Path
from ultralytics.yolo.utils import DEFAULT_CFG_PATH, ROOT, ops
from ultralytics.yolo.engine.results import Results

# ---------- 1. 读外部 YAML ----------
ext_cfg_path = Path(DEFAULT_CFG_PATH)          # 你的外部配置文件
with open(ext_cfg_path, encoding='utf-8') as f:
    ext_cfg = yaml.safe_load(f)

# 取出想要的字段（带默认值）
max_det     = ext_cfg.get('max_det', 300)
conf        = ext_cfg.get('conf', 0.25)
iou         = ext_cfg.get('iou', 0.7)
agnostic    = ext_cfg.get('agnostic', False)
classes     = ext_cfg.get('classes', None)

def postprocess(preds, img, orig_imgs, **kwargs):
    overrides = {
        "iou_thres": iou,
        "max_det": max_det,
        "agnostic" : agnostic,
        "classes": classes,
    }
    overrides.update(kwargs)

    preds = torch.from_numpy(preds).float().cpu()
    preds = ops.non_max_suppression(preds, **overrides)
    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        if not isinstance(orig_imgs, torch.Tensor):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        pred = pred.cpu().detach().numpy()
        results.append(pred)
    return results

def letterbox(img, new_shape=(640, 640), color=114, auto=True, stride=32):
    """
    等比例缩放 + 灰条填充，与 Ultralytics 保持一致
    :param img:        np.ndarray  BGR 原图
    :param new_shape:  (h, w)      目标尺寸
    :param color:      int         填充颜色
    :param auto:       bool        是否把尺寸对齐到 stride 倍数
    :param stride:     int         对齐基数
    :return: img_lb, ratio, (dw, dh)
    """
    shape = img.shape[:2]  # 当前 [h, w]

    # 1. 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 取较小值，保证长边≤目标

    # 2. 计算新尺寸（无灰条时）
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    # 3. 计算灰条像素
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 剩余像素

    if auto:  # 对齐到 stride 倍数，减少后面特征图对不齐的问题
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 取余数当灰条

    # 4. 缩放图像
    if shape[::-1] != new_unpad:  # 只有尺寸变化才 resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 5. 加灰条（上下、左右）
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img_lb = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return img_lb, (r, r), (dw, dh)

def preprocess(img:np.ndarray, new_shape:tuple=(640, 640), color:int=114, auto:bool=True, stride:int=32, type:np.dtype = np.float32):
    im_padded, _, _,  = letterbox(img, new_shape=new_shape, color=color, auto=auto, stride=stride)
    im_padded = im_padded.transpose(2, 0, 1)
    im_padded = im_padded.astype(type)
    im_padded /= 255.0
    im_padded = im_padded[np.newaxis, ...]
    return im_padded

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DEYOLO openvino inference *o*')
    parser.add_argument('--vi_input', default=r"D:\company_Tenda\35.DEYOLO\dataset\images\vis_train\00000.png", type=str, help='input vi image')
    parser.add_argument('--ir_input', default=r"D:\company_Tenda\35.DEYOLO\dataset\images\ir_train\00000.png", type=str, help='input ir image')
    parser.add_argument('--model_path', default=r"D:\ycx_git_repositories\DEYOLO_quantize\DEYOLO\runs\detect\train9\weights\FP32_openvino_model\model.xml", type=str, help='xml model path')
    parser.add_argument('--device', default='CPU', type=str, help='CPU, GPU, AUTO, MULTI:CPU,GPU, HETERO:CPU,GPU')
    parser.add_argument('--conf_thres', default=0.1, type=float, help='object confidence threshold')
    args = parser.parse_args()

    # ---- ① 加载 & 编译 ----
    core = ov.Core()
    model = core.read_model(args.model_path)
    compiled = core.compile_model(model, args.device)

    # ---- 打印输入输出节点名称 ----
    print("模型输入名称：")
    for input in model.inputs:
        print(input.get_any_name())
    for output in model.outputs:
        print(output.get_any_name())

    # ---- ② 获取输入形状 ----
    inp_shape_vi = compiled.input("images").shape  # [1,3,640,640] 举例
    inp_dtype_vi = compiled.input("images").get_element_type().to_dtype()
    inp_shape_ir = compiled.input("images").shape  # [1,3,640,640] 举例
    inp_dtype_ir = compiled.input("images").get_element_type().to_dtype()

    # ---- ③ 造随机输入（替换成你的图像预处理） ----
    im_vi = cv2.imread(args.vi_input, cv2.IMREAD_COLOR)
    im_vi_padded = preprocess(im_vi, new_shape=(inp_shape_vi[2], inp_shape_vi[3], inp_dtype_vi), auto=False, type=inp_dtype_ir)
    im_ir = cv2.imread(args.ir_input, cv2.IMREAD_COLOR)
    im_ir_padded = preprocess(im_ir, new_shape=(inp_shape_ir[2], inp_shape_ir[3], inp_dtype_ir), auto=False, type=inp_dtype_ir)

    # ---- ④ 同步推理 ----
    infer_req = compiled.create_infer_request()
    while True:
        t0 = time.perf_counter()  # 起点
        infer_req.infer({"images": im_vi_padded, "images1": im_ir_padded})  # key=0 或 input 名称均可
        ms = (time.perf_counter() - t0) * 1000
        print(f"耗时: {ms:.3f} ms")

    # # ---- ⑤ 取出输出 ----
    # out0 = infer_req.get_output_tensor(0).data  # NumPy ndarray
    #
    # results = postprocess(out0, im_vi_padded, [im_vi], conf_thres=0.1)
    #
    # for result in results:
    #     for out in result.astype(np.int32):
    #         x1, y1, x2, y2 = out[0:4]
    #         cv2.rectangle(im_vi, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    #
    # cv2.imshow("images", im_vi)
    # cv2.waitKey(0)
    # print(results)

