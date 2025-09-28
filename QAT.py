from torch.cuda import device

from ultralytics.yolo.v8.detect import DetectionTrainer

from ultralytics import YOLO
import torch.quantization
from DEYOLO_net import DEYOLO
# from PTQ_NNCF_implicitV2

if __name__ == '__main__':
    # 加载预训练模型
    yolo_model = YOLO(r'D:\ycx_git_repositories\DEYOLO_quantize\DEYOLO\runs\detect\train9\weights\best.pt', ycxNet=True, nc=6)
    # yolo_model = YOLO(r"DEYOLO.net", ycxNet=True, nc=6)
    # yolo_model = YOLO(r"D:\ycx_git_repositories\DEYOLO_quantize\DEYOLO\runs\detect\train9\weights\best.pt")
    model = yolo_model.model
    model.train()
    # 插入伪量化节点
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # model = torch.quantization.fuse_modules(model, [], inplace=True)
    torch.quantization.prepare_qat(model, inplace=True)

    # Train the model
    train_results = yolo_model.train(
        data="M3FD.yaml",  # path to dataset YAML
        epochs=1,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    torch.save(model.state_dict(), 'int8/fakequant_model.pt')

    # 转换为真正量化模型
    quantized_model = torch.quantization.convert(model.eval(), inplace=False)

    torch.save(model.state_dict(), 'int8/int8_model.pth')
