from ultralytics import YOLO

# Load the model
model = YOLO(r"D:\ycx_git_repositories\DEYOLO_quantize\DEYOLO\runs\detect\train9\weights\best.pt", ycxNet=True, nc = 6)

# Check the number of classes
print(model.names)

model.export(format='onnx', imgsz=640, batch=1, dynamic=False)


