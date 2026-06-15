import cv2
from ultralytics import YOLO

def main():
    # 1. Load your custom-trained face detection brain
    model_path = 'training/runs/my_face_model/experiment_1/weights/best.pt'
    print(f"Loading custom model from: {model_path}")
    model = YOLO(model_path)

    # 2. Set video_source = 0 for live video (webcam)

    video_source = 'IMG_4928.MOV'
    
    print(f"Connecting to iPhone camera stream at: {video_source}")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not connect to your phone's camera stream.")
        print("Double-check that your phone and PC are on the same Wi-Fi network and the app is open!")
        return

    print("Live Tracking active! Press 'q' on your keyboard to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Dropped frame or stream disconnected.")
            break

        # 3. Run the live frame through your custom face detector
        results = model(frame, conf=0.25, stream=True)

        # 4. Draw the bounding boxes
        annotated_frame = frame
        for r in results:
            annotated_frame = r.plot()

        # 5. Automatically scale the live window so it fits your monitor nicely
        orig_h, orig_w = annotated_frame.shape[:2]
        scale_factor = 0.4  # Shrinks the view window to 40% of full size
        
        display_width = int(orig_w * scale_factor)
        display_height = int(orig_h * scale_factor)
        resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

        # 6. Display the live tracked video
        cv2.imshow("Basel's Live iPhone Face Tracker", resized_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live tracker closed successfully.")

if __name__ == '__main__':
    main()