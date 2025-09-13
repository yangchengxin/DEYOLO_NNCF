from ultralytics import YOLO
import time

if __name__ == '__main__':
    # Load a model
    model = YOLO("runs/detect/train/weights/best.pt")  # trained weights

    results = model.val(
        data="M3FD.yaml",  # path to dataset YAML
        epochs=10,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    # # Perform object detection on RGB and IR image
    # start = time.time()
    # model.predict(["ultralytics/assets/vi_2.png", "ultralytics/assets/ir_2.png"],  # corresponding image pair
    #               save=True, imgsz=320, conf=0.1)
    # end = time.time()
    # cost_time = end - start
    # print(f"cost time: {cost_time}")
    #
    # model.val()