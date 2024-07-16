import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO
from YOLOv8InferenceClass import ObjectDetection
import time
from PIL import Image
import numpy as np
from detectK import DetectKnife

url_people = "../Model/count_people_ver2.pt"

# Create a Streamlit app
st.title("YOLOv8 People Counter App")

people_state = False
knife_state = False

option = st.selectbox(
    'How would you like to be contacted?',
    ('Video File', 'Camera', 'Image'))

option_model = st.multiselect(
    'Select Objects to Detect:',
    ('Knife', 'People'))

if 'Knife' in option_model and 'People' in option_model:
    # Both Knife and People selected
    people_state = True
    knife_state = True
elif 'Knife' in option_model:
    # Only Knife selected
    knife_state = True
elif 'People' in option_model:
    # Only People selected
    people_state = True
else:
    st.warning("Please select at least one object to detect.")


if knife_state:
    confidence_threshold_knife = st.slider('Select Confidence Threshold for **Knife** Detection:', 0.0, 1.0, 0.4)
if people_state:
    confidence_threshold_people = st.slider('Select Confidence Threshold for **People** Detection:', 0.0, 1.0, 0.4)
# model
model_people = ObjectDetection(url_people, confidence_threshold_people)
mod = YOLO("../Model/SmallKnife.pt")

def calAndDraw(frame, tex):
    
    frame_height, frame_width, _ = frame.shape
    text_width, text_height = cv2.getTextSize(tex, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    
    # Tính toán tọa độ x và y cho văn bản và hộp đen
    text_x = frame_width - text_width - 10  # 10 là một khoảng cách từ mép phải của frame
    text_y = 30  # Vị trí y cố định
    box_x1 = text_x - 5
    box_y1 = text_y - text_height - 5
    box_x2 = frame_width - 5
    box_y2 = text_y + 5

    # Vẽ hộp đen
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.putText(frame,tex, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def calAndDraw2(frame, tex):
    
    frame_height, frame_width, _ = frame.shape
    text_width, text_height = cv2.getTextSize(tex, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    
    # Tính toán tọa độ x và y cho văn bản và hộp đen
    text_x = 10  # 10 là một khoảng cách từ mép trái của frame
    text_y = 30  # Vị trí y cố định
    box_x1 = text_x - 5
    box_y1 = text_y - text_height - 5
    box_x2 = text_x + text_width + 5
    box_y2 = text_y + 5

    # Vẽ hộp đen
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.putText(frame,tex, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

video_source = None
# video_file = None
if option == 'Camera':
    video_source = 0
    video_file = 1
elif option == 'Video File':
    video_file = st.file_uploader("Upload a video file", type=["mp4"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_source = tfile.name
elif option =='Image':
    image_file = st.file_uploader('Uploade an image', type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if people_state:
            # frame = countPeople.countPeople(image_np, model_people, confidence_threshold_people)
            frame, results1 = model_people.run(frame = image_np)
        
        
        if knife_state:
            # frame, results2 = model_knife.run(frame= image_np)
            try:
                frame, c = DetectKnife(model = mod, results = results1, frame = frame, conf = confidence_threshold_knife, cou = c).detect()
            except:
                frame, c = DetectKnife(model = mod, results = results1, frame = frame, conf = confidence_threshold_knife, cou = 0).detect()

        # frame = calAndDraw(frame)
        frame_placeholder = st.empty()
        frame_placeholder.image(frame, channels="BGR")

def plot_knife(kq, frame):
    for ans in kq:
        if ans ==0:
            continue
        for a in ans:
            x1, y1, x2, y2 = a[0], a[1], a[2], a[3]
            frame = cv2.rectangle(frame, (x1,y1), (x2, y2), color= (0, 0, 255), thickness=10)
            # cv2.imwrite(f"image/{self.count}.jpg", self.FRAME)
            # self.count +=1
    return frame

    
if video_source is not None and (people_state or knife_state):
    # print(video_source)
    cap = cv2.VideoCapture(video_source)
    print('Opened', cap)
    frame_placeholder = st.empty()

    frame_count = 0
    start_time = time.time()
    con = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_count += 1
            end_time = time.time()
            time_per_frame = (end_time - start_time) * 1000  # Đổi sang ms
            start_time = time.time()
            
            frame_save = frame.copy()
            if people_state:
                frame, results1 = model_people.run(frame= frame)
            if len(results1[0].boxes.cls) != 0:
                try:
                    re, c = DetectKnife(model = mod, results = results1, frame = frame_save, conf = confidence_threshold_knife, cou = c).detect()
                except:
                    re, c = DetectKnife(model = mod, results = results1, frame = frame_save, conf = confidence_threshold_knife, cou = 0).detect()

                frame = plot_knife(re, frame)
                
            num_people = len(results1[0].boxes.cls)
            tex2 = f'Person: {num_people}'
            tex = f'{time_per_frame:.2f} ms'
            frame = calAndDraw(frame,tex)
            frame = calAndDraw2(frame, tex2)
            frame_placeholder.image(frame, channels="BGR")
        else:
            break
    frame_placeholder = st.empty()
    cap.release()
    

