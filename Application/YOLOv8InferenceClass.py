import torch
import numpy as np
from time import time
from ultralytics import YOLO
import supervision as sv

class ObjectDetection:

    def __init__(self, model1, conf):
        self.model1 = model1
        self.CONF = conf
        # self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=5, text_thickness=1, text_scale=1)
    

    def load_model(self):
        model = YOLO(self.model1)  # load a pretrained YOLOv8n model
        return model


    def predict(self, frame):
        results = self.model(frame, conf = self.CONF, iou=0.8)
        return results
    

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            class_id = boxes.cls[0]
            conf = boxes.conf[0]
            xyxy = boxes.xyxy[0]

            if class_id == 0.0:
          
              xyxys.append(result.boxes.xyxy.cpu().numpy())
              confidences.append(result.boxes.conf.cpu().numpy())
              class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        # Setup detections for visualization
        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections]
        
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        # frame = self.box_annotator.annotate(scene=frame, detections=detections,  labels= None)
        
        return frame
        
    def run(self, frame):
        results = self.predict(frame)
        try:
            frame = self.plot_bboxes(results, frame)
        except:
            pass
        return frame, results
    
