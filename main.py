from ultralytics import YOLO


# Load a model
#model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-pose.pt')

  # train the model

results = model.train(data='mydata.yaml', epochs=50, save=True)  
