from ultralytics import YOLO

# Load the model
model = YOLO(r"D:\ycx_git_repositories\DEYOLO_NNCF\DEYOLO\M3FD_best.pt")

# Check the number of classes
print(model.names)

model.export(format='onnx', imgsz=640, batch=1, dynamic=False)


