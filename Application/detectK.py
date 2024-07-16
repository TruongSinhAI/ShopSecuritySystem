import cv2 
from ultralytics import YOLO 

class DetectKnife:
    def __init__(self, model, results, frame, conf, cou) -> None:
        self.MODEL = model
        self.results = results
        self.kq = []
        self.FRAME  = frame
        self.CONF = conf
        self.count  = cou
        
    def listObject(self):
        lsFrame = []
        for result in self.results:
            boxes = result.boxes.cpu().numpy()
            try:
                x1, y1, x2, y2 = map(int, boxes.xyxy[0])
                lsFrame.append((x1,y1,x2,y2))
            except:
                pass
        return lsFrame
    
    def detect(self):
        lsFrame = self.listObject()
        if len(lsFrame) ==0:
            return self.FRAME
        for frame in lsFrame:
            if len(frame) ==0:
                continue
            try:
                ans = self.MODEL(self.FRAME[frame[1]-50:frame[3]+50, frame[0]-50:frame[2]+50], conf = self.CONF, iou=0.8)
            except:
                continue
            if len(ans[0].boxes.cls) == 0:
                self.kq.append(0)
            else:
                x = []
                cv2.imwrite(f"image/{self.count}.jpg", self.FRAME[frame[1]-50:frame[3]+50, frame[0]-50:frame[2]+50])
                self.count +=1
                for a in ans:
                    boxes = a.boxes.cpu().numpy()
                    x1, y1, x2, y2 = map(int, boxes.xyxy[0])
                    x.append([frame[0]-50+x1, frame[1]-50+y1,frame[2]+50-x2, frame[3]+50-y2])
                    
                self.kq.append(x)
        
        return self.kq, self.count
