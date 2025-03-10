import webcolors
import cv2
import numpy as np
import statistics

def get_colour_from_subimage(key, tracks_current, img):
    for track in tracks_current:
        if(track.track_id != key):
            continue;
        print(track.track_id)
        location = track.to_tlbr()
        print(location)
        bbox = location[:4].astype(int)
        # format is top left xy, bottom right xy
        subimage = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        median_blue = cv2.calcHist([subimage], [0], None, [256], [0,256])
        print("MEDIAN BLUE: " + str(median_blue))
        median_green = cv2.calcHist([subimage], [1], None, [256], [0,256])
        print("MEDIAN GREEN: " + str(median_green))
        median_red = cv2.calcHist([subimage], [2], None, [256], [0,256])
        print("MEDIAN BLUE: " + str(median_red))
        #TODO: look at these links https://answers.opencv.org/question/20522/get-the-median-added-to-mean-and-std-value/ https://stackoverflow.com/questions/23255903/finding-the-median-value-of-an-rgb-image-in-opencv https://www.geeksforgeeks.org/python-opencv-cv2-calchist-method/
