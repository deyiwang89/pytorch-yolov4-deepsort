from .YOLOV4 import yolo
from .YOLOV4.yolo import YOLO


__all__ = ['build_detector']

def build_detector():
    return YOLO()
