# -*- coding: UTF-8 -*-
"""
  @Author: ycx
  @Date  : 2025/9/8 16:26
  @Email : <ycx97124@163.com>
  @version V1.0
"""

import cv2
import torch
import requests
import numpy as np
import openvino as ov
from pathlib import Path
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.v8.detect import DetectionPredictor
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.engine.results import Results


class OV_DetectionPredictor(BasePredictor):
    def __init__(self, args, model_names, device='CPU'):
        super().__init__(args)
        self.device = device
        self.model_names = model_names
        self.input_size = (args.imgsz, args.imgsz)  # 假设正方形

    def postprocess(self, preds, img, orig_imgs):
        preds = torch.from_numpy(preds).to(self.device, dtype=torch.float32)
        preds = ops.non_max_suppression(
            preds, self.args.conf, self.args.iou,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
            classes=self.args.classes)

        results = []
        orig_imgs = orig_imgs if isinstance(orig_imgs, list) else [orig_imgs]
        paths = self.batch[0] if isinstance(self.batch[0], list) else [self.batch[0]]

        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            if len(pred):
                pred[:, :4] = ops.scale_boxes(self.input_size, pred[:, :4], orig_img.shape)
            results.append(Results(orig_img=orig_img,
                                   path=paths[i],
                                   names=self.model_names,
                                   boxes=pred))
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
    model_path = r"D:\ycx_git_repositories\DEYOLO_NNCF\DEYOLO\runs\detect\train\weights\INT8_openvino_model\model.xml"  # 同目录下要求有 best.bin
    device = "CPU"  # 或 "GPU" / "AUTO"

    # ---- ① 加载 & 编译 ----
    core = ov.Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(model, device)

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
    im_vi = cv2.imread(r"D:\company_Tenda\35.DEYOLO\dataset\images\ir_train\00000.png", cv2.IMREAD_COLOR)
    im_vi_padded = preprocess(im_vi, new_shape=(inp_shape_vi[2], inp_shape_vi[3], inp_dtype_vi), auto=False, type=inp_dtype_ir)
    im_ir = cv2.imread(r"D:\company_Tenda\35.DEYOLO\dataset\images\ir_train\00000.png", cv2.IMREAD_COLOR)
    im_ir_padded = preprocess(im_ir, new_shape=(inp_shape_ir[2], inp_shape_ir[3], inp_dtype_ir), auto=False, type=inp_dtype_ir)

    # ---- ④ 同步推理 ----
    infer_req = compiled.create_infer_request()
    infer_req.infer({"images": im_vi_padded, "images1": im_ir_padded})  # key=0 或 input 名称均可

    # ---- ⑤ 取出输出 ----
    out0 = infer_req.get_output_tensor(0).data  # NumPy ndarray

    results = OV_DetectionPredictor().postprocess(out0, im_vi_padded, [im_vi])
    print(results)

