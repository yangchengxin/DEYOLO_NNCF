from ultralytics import YOLO
import time as time

# Load a model
model = YOLO("runs/detect/train/weights/best.pt") # trained weights

# Perform object detection on RGB and IR image
start = time.time()
model.predict(["ultralytics/assets/vi_2.png", "ultralytics/assets/ir_2.png"], # corresponding image pair
              save=True, imgsz=320, conf=0.1)
end = time.time()
cost_time =end - start
print(f"cost time: {cost_time}")