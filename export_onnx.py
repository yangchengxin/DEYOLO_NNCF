from ultralytics import YOLO

# Load the model
model = YOLO(r"D:\company_Tenda\35.DEYOLO\DEYOLO\runs\detect\train\weights\best.pt")

# Check the number of classes
print(model.names)

model.export(format='onnx', imgsz=640, batch=1, dynamic=False)
