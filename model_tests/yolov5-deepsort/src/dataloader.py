import cv2
import yaml
from interfaces import Input

with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['yolov5_deepsort']['dataloader']

# Data Source Parameters
DATA_SOURCE = config['data_source']   
WEBCAM_ID = config['webcam_id']  
DATA_PATH = config['data_path']  
FRAME_WIDTH = config['frame_width']
FRAME_HEIGHT = config['frame_height'] 

# # Select Data Source
# if DATA_SOURCE == "webcam":
#     cap = cv2.VideoCapture(WEBCAM_ID)
# elif DATA_SOURCE == "video file":
#     cap = cv2.VideoCapture(DATA_PATH)
# else:
#     print("Enter correct data source")

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

class StaticInput(Input):
    def __init__(self, data_path, frame_width, frame_height):
        self.cap = cv2.VideoCapture(data_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

class CamInput(Input):
    def __init__(self, webcam_id, frame_width, frame_height):
        self.cap = cv2.VideoCapture(webcam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

def construct_input(data_source, data_info, frame_width, frame_height) -> Input:
    if data_source == "webcam":
        return CamInput(data_info, frame_width, frame_height)
    elif data_source == "video file":
        return StaticInput(data_info, frame_width, frame_height)
    else:
        raise AttributeError(f"data_source is invalid: {data_source =}")
