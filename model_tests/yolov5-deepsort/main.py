import cv2
import time
import os 
import sys
import yaml
import webcolors

from keras.models import load_model

from src.detector import YOLOv5Detector
from src.tracker import DeepSortTracker
from src.dataloader import cap
from src.colour_getter import get_colour_from_subimage
from src.type_identifier import identify_vehicle_type

colour_names = webcolors.names(webcolors.CSS3)
colour_codes = []
for colour in colour_names:
    colour_codes.append(webcolors.name_to_hex(colour))

colour_dict = list(zip(colour_names, colour_codes)) 
#needed for the colour getting. This is declared here because python doesn't have constants so it's passed in

# Parameters from config.yml file
with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['yolov5_deepsort']['main']

# Add the src directory to the module search path
sys.path.append(os.path.abspath('src'))

#Load the identification model from https://github.com/hoanhle/Vehicle-Type-Detection

identification_model = load_model('./src/mobilenet2.h5')

identification_dictionary = dict(zip([i for i in range(17)], ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']))

# Get YOLO Model Parameter
YOLO_MODEL_NAME = config['model_name']

print("test")

# Visualization Parameters
DISP_FPS = config['disp_fps'] 
DISP_OBJ_COUNT = config['disp_obj_count']

object_detector = YOLOv5Detector(model_name=YOLO_MODEL_NAME)
tracker = DeepSortTracker()

track_history = {}    # Define a empty dictionary to store the previous center locations for each track ID

objects_no_longer_in_scene = {}

object_start_frame = {}

object_end_frame = {}

track_frame_length = {}

frame_count = 1;

vehicle_type = {};

vehicle_colour = {};

while cap.isOpened():

    success, img = cap.read() # Read the image frame from data source 
 
    start_time = time.perf_counter()    #Start Timer - needed to calculate FPS
    
    # Object Detection
    print("test")
    results = object_detector.run_yolo(img)  # run the yolo v5 object detector 
    
    #TODO: Maybe put in here a check to see if an object is new and to start counting its frames

    detections , num_objects= object_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[1]) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected

    print("DETECTIONS:\n" + str(detections))

    #results is a tuple
    #num_objects is an int
    #detections is a list
    #tracks_current is a list

#    print("type of results " + str(results))
#    print("type of num_objects " + str(num_objects))
#    print("type of detections " + str(detections))
#
    # Object Tracking

    tracks_current = tracker.object_tracker.update_tracks(detections, frame=img)
    tracker.display_track(track_history , tracks_current , img)

    #TODO: get the subimage defined by the bounding boxes from the tracker/detector
    # Pass the subimage to the vehicle classifier model and the average colour subroutine - ONCE

    to_be_destroyed = []

    if(frame_count % 2 == 0):
    
        for key in track_history:
            if(not any(key == value.track_id for value in tracks_current)): #if the key has left the scene
                to_be_destroyed.append(key)
            elif key not in object_start_frame:
                object_start_frame[key] = frame_count
            elif((key in object_end_frame) & (key not in vehicle_colour)):
                to_be_destroyed.append(key)
    
        for key in to_be_destroyed: #deal with the tracks which have left the scene
            objects_no_longer_in_scene[key] = track_history.get(key, [])
            del track_history[key]
            object_end_frame[key] = frame_count
            if((key in object_end_frame) & (key not in vehicle_colour)):
                del object_start_frame[key]
                del object_end_frame[key]
                del objects_no_longer_in_scene[key]
                # it shouldn't be in vehicle_colour or vehicle_type

        #print(type(tracks_current[0].track_id))
        
        #TODO: get the subimage defined by the bounding boxes from the tracker/detector
        # Pass the subimage to the vehicle classifier model and the average colour subroutine - ONCE
    
        for key in track_history: #should have got rid of the ones not in the scene
            if(key not in vehicle_colour):
                if((frame_count - object_start_frame[key] > 3) & (key not in object_end_frame)):
                    print("detecting vehicle type for " + str(key));
                    vehicle_colour_local, subimage = get_colour_from_subimage(key, tracks_current, img, colour_dict, frame_count) 
                    if(vehicle_colour_local == "AGAIN"):
                        continue
                    vehicle_colour[key] = vehicle_colour_local
                    vehicle_type[key] = identify_vehicle_type(subimage, identification_model, identification_dictionary) #TODO: identifier is not the most accurate, could need further training
    
        #DEBUG CODE BELOW HERE
        for key in objects_no_longer_in_scene:
            print("Car ID " + key + " entered at frame: " + str(object_start_frame[key]) + " and left at frame: " +  str(object_end_frame[key]) + " with colour: " + str(vehicle_colour[key]) + " and type: " + str(vehicle_type[key]))

        print("THE CARS CURRENTLY IN THE SCENE ARE: " + str([track.track_id for track in tracks_current]))
        print(objects_no_longer_in_scene)
        #END

    
    # FPS Calculation
    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = 1 / total_time


    # Descriptions on the output visualization
    cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, f'MODEL: {YOLO_MODEL_NAME}', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, f'TRACKED CLASS: {object_detector.tracked_class}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, f'TRACKER: {tracker.algo_name}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, f'DETECTED OBJECTS: {num_objects}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    cv2.imshow('img',img)

    


    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1


# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
