from ultralytics import YOLO
model = YOLO('yolov8n.pt','v8')  # load a pretrained model (recommended for training)

#pridict on an image
detection_output = model.predict(source='https://ultralytics.com/images/bus.jpg',conf=0.25,save=True)


# display the tensor array of the detection output
print(detection_output)


# display the detection output in a more readable format
print(detection_output[0].numpy())  # print the bounding box data