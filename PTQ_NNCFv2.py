# -*- coding: UTF-8 -*-
"""
  @Author: ycx
  @Date  : 2025/9/8 13:44
  @Email : <ycx97124@163.com>
  @version V1.0
"""


"""本文件用于将openvino的32位权重文件后训练量化为INT8"""
# 设定关闭FP量化实验功能，使用较为稳定的INT8
import os
os.environ["NNCF_DISABLE_F8_IMPLEMENTATION"] = "1"

# 导入openvino库的运行模块
from openvino.runtime import Core, serialize  # serialize用于将量化的模型保存为ir文件，并输出xml,bin文件

# 导入nncf库的量化对象，数据集对象
from nncf import quantize, Dataset
from nncf.quantization import QuantizationPreset  # 修复导入

import numpy as np
import cv2
from tqdm import tqdm  # 进度条
from pathlib import Path  # 处理路径
import nncf

DATA_PATH_VIS = r"D:\company_Tenda\35.DEYOLO\DEYOLO\dataset\images\vis_train"  # 就用训练集数据校准，也可以用其他数据
DATA_PATH_IR = r"D:\company_Tenda\35.DEYOLO\DEYOLO\dataset\images\ir_train"  # 就用训练集数据校准，也可以用其他数据
NUM_CALIB = 300  # 矫正数据集常量,和文件夹中图像数据一致


# 3. 创建校准数据集类
class CalibrationDataset:
    def __init__(self, data_path_vis=DATA_PATH_VIS, data_path_ir = DATA_PATH_IR, num_samples=300):

        self.data_path_vis = Path(data_path_vis)
        self.data_path_ir = Path(data_path_ir)
        self.num_samples = num_samples

        # 检查是否有真实图像可用，如果要求模型准确，
        if self.data_path_vis.exists() and len(list(self.data_path_vis.glob("*.*"))) > 0 and self.data_path_ir.exists() and len(list(self.data_path_ir.glob("*.*"))) > 0:
            self.image_files_vis = list(self.data_path_vis.glob("*.*"))[:num_samples]
            self.image_files_ir = list(self.data_path_ir.glob("*.*"))[:num_samples]
            self.use_real_images = True
            print(f"使用真实校准图像: {len(self.image_files_vis)}张")
        else:
            self.use_real_images = False
            print(f"使用随机生成数据: {num_samples}张")

    def __getitem__(self, index):
        if self.use_real_images:
            # 从真实图像加载
            img_path_vis = self.image_files_vis[index % len(self.image_files_vis)]
            img_vis = cv2.imread(str(img_path_vis))
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

            img_path_ir = self.image_files_ir[index % len(self.image_files_ir)]
            img_ir = cv2.imread(str(img_path_ir))
            img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)
        else:
            # 生成随机图像
            img_vis = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
            img_ir = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

        # 预处理
        img_vis = cv2.resize(img_vis, (640, 640))
        img_vis = img_vis.transpose(2, 0, 1)  # HWC to CHW
        img_vis = img_vis[np.newaxis, ...]  # 添加批次维度

        img_ir = cv2.resize(img_ir, (640, 640))
        img_ir = img_ir.transpose(2, 0, 1)  # HWC to CHW
        img_ir = img_ir[np.newaxis, ...]  # 添加批次维度
        return {"images": img_vis.astype(np.float32) / 255.0, "images1": img_ir.astype(np.float32) / 255.0}


# INT量化神经网络
def quantize_to_int8():
    print("开始INT8量化...")
    # 实例化openvino的核心模块并加载模型
    core = Core()
    model = core.read_model(r"D:\company_Tenda\35.DEYOLO\DEYOLO\runs\detect\train\weights\FP32_openvino_model\model.xml")
    # 创建校准数据集
    calibration_dataset = CalibrationDataset(num_samples=NUM_CALIB)

    # 使用NNCF Dataset包装器
    nncf_dataset = nncf.Dataset(calibration_dataset)

    # 执行量化
    # -- 预设值	含义	适用场景
    # -- PERFORMANCE	优先推理速度和模型压缩率	部署到边缘设备、实时推理
    # -- ACCURACY	优先保留模型精度	精度敏感任务，如医疗、金融
    # -- MIXED	在精度和性能之间折中	通用部署，适合大多数场景
    quantized_model = quantize(
        model=model,
        calibration_dataset=nncf_dataset,
        preset=QuantizationPreset.PERFORMANCE
    )

    # 保存量化模型
    serialize(quantized_model,
              "openvino_model/DEYOLO_int8.xml",
              "openvino_model/DEYOLO_int8.bin")
    print("INT8量化完成! 模型保存在 openvino_model/ 目录")


if __name__ == "__main__":
    # # 创建输出目录
    # os.makedirs("openvino_model", exist_ok=True)  # os.makedirs()创建多及目录
    #   
    # quantize_to_int8()

    from ultralytics import YOLO

    ov_model = YOLO(r"D:\company_Tenda\35.DEYOLO\DEYOLO\_openvino_model")
    metrics = ov_model.val(data=r"D:\company_Tenda\35.DEYOLO\DEYOLO\ultralytics\yolo\cfg\M3FD.yaml")
    print('mAP50:', metrics.box.map50)
    print('mAP50-95:', metrics.box.map)