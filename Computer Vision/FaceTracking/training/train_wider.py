import os
from ultralytics import YOLO

def main():
    # Load the base model
    model = YOLO('yolov8n.pt') 

    # Train the model using the updated YAML file
    model.train(
        data='../wider_face.yaml', 
        epochs=10, 
        imgsz=640, 
        device=0,               # Set as integer 0 to cleanly target your RTX 5070
        project='../my_face_model', # Saves the model weights up in the main FaceTracking folder
        name='experiment_1',
        save=True
    ) 

if __name__ == '__main__':
    main()