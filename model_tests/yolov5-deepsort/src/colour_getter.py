import webcolors
import cv2
from colourconvert import rgb2hex
import numpy as np
import math
from sklearn.cluster import KMeans


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

def get_colour_from_subimage(key, tracks_current, img, colour_dict): #also returns the subimage for later passing into the vehicle type detection thing
    for track in tracks_current:
        if(track.track_id != key):
            continue;
        #print(track.track_id)
        location = track.to_tlwh(orig=True)
        #print("KEY: " + str(key))
        #print(location)

        bbox = location[:4].astype(int)

        bbox2 = [int(x) for x in bbox]
        # format is top left xy, width, height
        if(bbox2[0] < 0 | bbox2[1] < 0):
            print("AGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\n") #debug code, can be removed
            return "AGAIN", None
        if(bbox[3] > 45 and bbox[2] > 45): #This one determines the minimum size of the bounding box before it checks the colour, can be messed with
            #print("AGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\nAGAIN\n") #debug code, can be removed
            return "AGAIN", None
        else:
            print(bbox[0])
            print(bbox[1])
            print("this hasn't worked\n")
        subimage = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

        print(bbox)
 
        #if(key == "2"):
            #print("BRUH BRUH BRUH")
            #print(str(bbox[0]) + str(bbox[2]) + str(bbox[1]) + str(bbox[3]))
            #cv2.imshow("subimage", subimage)

        #apply preprocessing

        blurred_image = cv2.GaussianBlur(subimage, (1, 1), 0)
        
        #subimage_hsv = cv2.cvtColor(subimage, cv2.COLOR_BGR2HSV)

        kmeans = KMeans(n_clusters=4)

        #resized_image = cv2.resize(blurred_image, (100, 100), interpolation = cv2.INTER_AREA)

        reshaped = blurred_image.reshape((-1, 3)) #puts it in the proper format for the kmeans clustering

        #fit the clustering

        kmeans.fit(reshaped)

        #find the largest cluster

        labels = kmeans.labels_

        unique, counts = np.unique(labels, return_counts=True)

        cluster_sizes = dict(zip(unique, counts))

        largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)

        largest_cluster_centre = kmeans.cluster_centers_[largest_cluster_label]
        #this should be a bgr value?


        #print(largest_cluster_centre)
        #print(kmeans.cluster_centers_)
        #print(cluster_sizes)

        largest_cluster_centre = [int(x) for x in largest_cluster_centre.tolist()]
        
        print(largest_cluster_centre)

        car_colour = get_colour_name(webcolors.rgb_to_hex(tuple(largest_cluster_centre)), colour_dict)

        print("THE CAR WITH ID " + str(track.track_id) + " IS COLOURED " + car_colour)

        return car_colour, subimage

        #TODO: look at these links https://answers.opencv.org/question/20522/get-the-median-added-to-mean-and-std-value/ https://stackoverflow.com/questions/23255903/finding-the-median-value-of-an-rgb-image-in-opencv https://www.geeksforgeeks.org/python-opencv-cv2-calchist-method/
