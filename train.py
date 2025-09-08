from ultralytics import YOLO

if __name__ == '__main__':
    # ------------------------------------ train ------------------------------ #
    # Load a model
    model = YOLO("ultralytics/models/v8/DEYOLO.yaml").load("yolov8n.pt")

    # Train the model
    train_results = model.train(
        data="M3FD.yaml",  # path to dataset YAML
        epochs=1,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )
    # ------------------------------------ train ------------------------------ #

