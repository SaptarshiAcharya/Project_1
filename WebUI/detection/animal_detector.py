from ultralytics import YOLO
import cv2


class AnimalDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_animals(self, frame):
        """Run YOLO detection and return annotated frame + detected animals."""
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()

        detected_animals = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detected_animals.append((self.model.names[cls], conf))

        return annotated_frame, detected_animals
