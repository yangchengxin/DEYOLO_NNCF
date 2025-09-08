import os
import nncf
import torch
import openvino
from nncf import quantize
from nncf import compress_weights
from ultralytics import YOLO

def create_data_source():
    from ultralytics.yolo.engine.trainer import BaseTrainer
    from ultralytics.yolo.v8.detect.train import DetectionTrainer
    from ultralytics.yolo.utils import DEFAULT_CFG

    cfg = DEFAULT_CFG
    model = "ultralytics/models/v8/DEYOLO.yaml"
    data = cfg.data or "ultralytics/yolo/cfg/M3FD.yaml"  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    trainer = DetectionTrainer(overrides=DEFAULT_CFG)
    trainer.setup_model()
    vis_dir = r"D:\company_Tenda\35.DEYOLO\DEYOLO\dataset\images\vis_train"
    inf_dir = r"D:\company_Tenda\35.DEYOLO\DEYOLO\dataset\images\ir_train"
    dataloader = trainer.get_dataloader(
        dataset_path = vis_dir,
        dataset_path2 = inf_dir,
        batch_size = 1,
        rank = -1,
        mode = 'train'  # 或 'val'
    )
    return dataloader


def transform_fn(batch):
    batch_input = dict()
    # 网络的输入节点名称是images和images1
    batch_input['images'] = batch['img'].to(torch.device("cpu"), non_blocking=True).float() / 255
    batch_input['images1'] = batch['img2'].to(torch.device("cpu"), non_blocking=True).float() / 255
    return batch_input

#代码需要放在main下执行，否则会报进程错误：
#这是一个关于windows上多进程实现的恩特。在windows上，子进程会自动import启动它的这个文件，而在import的时候是会自动执行这些语句的。
#如果不加__main__限制的化，就会无限递归创建子进程，进而报错。于是import的时候使用 name == “main” 保护起来就可以了。
if __name__ == "__main__":
    data_yaml = r"D:\company_Tenda\35.DEYOLO\DEYOLO\ultralytics\yolo\cfg\M3FD.yaml"
    MODEL_PATH = r"D:\company_Tenda\35.DEYOLO\DEYOLO\runs\detect\train\weights\best.onnx"
    FP32_dir = r"D:\company_Tenda\35.DEYOLO\DEYOLO\runs\detect\train\weights\FP32_openvino_model"
    FP16_dir = r"D:\company_Tenda\35.DEYOLO\DEYOLO\runs\detect\train\weights\FP16_openvino_model"
    INT8_dir = r"D:\company_Tenda\35.DEYOLO\DEYOLO\runs\detect\train\weights\INT8_openvino_model"

    for it in [FP32_dir, FP16_dir, INT8_dir]:
        if not os.path.exists(it):
            os.makedirs(it)
            print(f"已创建文件夹：{it}")
        else:
            print(f"文件夹已存在：{it}")

    # # ----------------------------------------- 查看xml的输入输出节点 ----------------------------------------- #
    from openvino.runtime import Core
    ie = Core()
    model = ie.read_model(r"D:\company_Tenda\35.DEYOLO\DEYOLO\runs\detect\train\weights\FP16_openvino_model\model.xml")
    print("模型输入名称：")
    for input in model.inputs:
        print(input.get_any_name())
    for output in model.outputs:
        print(output.get_any_name())

    # # ----------------------------------------- 导出fp32，fp16，int8三种精度的模型，其中int8使用的是校准数据集的方式 ----------------------------------------- #
    # fp32 IR model
    print(f"Export ONNX to Openvino FP32 IR to:{FP32_dir}")
    model_fp32 = openvino.convert_model(MODEL_PATH)
    openvino.save_model(model_fp32, f"{FP32_dir}/model.xml", compress_to_fp16=False)

    #fp16 IR model
    print(f"Export onnx to openvino FP16 IR to:{FP16_dir}")
    model_fp16 = openvino.convert_model(MODEL_PATH)
    openvino.save_model(model_fp16, f"{FP16_dir}/model.xml", compress_to_fp16=True)

    #int8 IR model
    print(f"Export onnx to openvino INT8 IR to:{INT8_dir}")
    data_source = create_data_source()
    nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)
    model_int8 = openvino.convert_model(MODEL_PATH)
    quantized_model = quantize(
        model_int8,
        nncf_calibration_dataset,
        subset_size=10,  # 使用1000个样本进行校正
    )
    openvino.save_model(quantized_model, f"{INT8_dir}/model.xml", compress_to_fp16=False)

    # ----------------------------------------- 导出int8精度的模型，使用直接映射的方式 ----------------------------------------- #

    # 待补充

    # ----------------------------------------- validate ----------------------------------------- #
    ov_model = YOLO(r"D:\company_Tenda\35.DEYOLO\DEYOLO\runs\detect\train\weights\INT8_openvino_model")
    metrics = ov_model.val(data=r"D:\company_Tenda\35.DEYOLO\DEYOLO\ultralytics\yolo\cfg\M3FD.yaml")
    print('mAP50:', metrics.box.map50)
    print('mAP50-95:', metrics.box.map)

