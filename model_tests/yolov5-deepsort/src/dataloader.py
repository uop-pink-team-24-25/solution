import cv2
import yaml
from src.interfaces import Input

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

def construct_input(data_source, webcam_id, file, frame_width, frame_height) -> Input:
    if data_source == "webcam":
        return CamInput(webcam_id, frame_width, frame_height)
    elif data_source == "video file":
        return StaticInput(file, frame_width, frame_height)
    else:
        raise AttributeError(f"data_source is invalid: {data_source =}")
