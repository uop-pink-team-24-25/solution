import webcolors
import cv2
import rgb2hex
import numpy as np
import math


def calc_median(histogram):
    histogram.sort()
    return histogram[len(histogram) // 2]

def get_approx_colour(hex_colour, colour_dict):
    orig = webcolors.hex_to_rgb(hex_colour)
    similarity = {}
    for colour_name, hex_code in colour_dict:
        approx = webcolors.hex_to_rgb(hex_code)
        similarity[colour_name] = sum(np.subtract(orig, approx) ** 2)
    return min(similarity, key=similarity.get)
 
def get_colour_name(hex_colour, colour_dict):
    try:
        return webcolors.hex_to_name(hex_colour)
    except ValueError:
        return get_approx_colour(hex_colour, colour_dict)

def get_colour_from_subimage(key, tracks_current, img, colour_dict):
    for track in tracks_current:
        if(track.track_id != key):
            continue;
        #print(track.track_id)
        location = track.to_tlbr()
        #print(location)
        bbox = location[:4].astype(int)
        # format is top left xy, bottom right xy
        subimage = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        histogram_blue = cv2.calcHist([subimage], [0], None, [256], [0,256])
        histogram_blue = [x for x in histogram_blue if x != 0]
        median_blue = np.mean(histogram_blue)
        print("MEDIAN BLUE: " + str(median_blue))

        histogram_green = cv2.calcHist([subimage], [1], None, [256], [0,256])
        histogram_green = [x for x in histogram_green if x != 0]
        median_green = np.mean(histogram_green)
        print("MEDIAN GREEN: " + str(median_green))
        
        histogram_red = cv2.calcHist([subimage], [2], None, [256], [0,256])
        histogram_red = [x for x in histogram_red if x != 0]
        #print("HISTOGRAM RED SIZE: " + str(max(histogram_red)))
        median_red = np.mean(histogram_red)
        print("MEDIAN RED: " + str(median_red))

        if(math.isnan(median_red)):
            median_red = 0

        if(math.isnan(median_green)):
            median_green = 0

        if(math.isnan(median_blue)):
            median_blue = 0

        car_colour = get_colour_name(webcolors.rgb_to_hex((int(median_red), int(median_green), int(median_blue))), colour_dict)

        print("THE CAR WITH ID " + str(track.track_id) + " IS COLOURED " + car_colour)

        return car_colour

        #TODO: look at these links https://answers.opencv.org/question/20522/get-the-median-added-to-mean-and-std-value/ https://stackoverflow.com/questions/23255903/finding-the-median-value-of-an-rgb-image-in-opencv https://www.geeksforgeeks.org/python-opencv-cv2-calchist-method/
