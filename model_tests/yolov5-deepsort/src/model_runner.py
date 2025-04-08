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


class ai_model(object):

    def __init__(self, config_path):
        self.__identification_model = load_model('./src/mobilenet2.h5')

        self.__identification_dictionary = dict(zip([i for i in range(17)], ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']))

        self.__DISP_FPS = None
        self.__DISP_OBJECT_COUNT = None
        self.config = None
        self.object_detector = None
        self.tracker = None
        self.track_history = {}    # Define a empty dictionary to store the previous center locations for each track ID
        self.objects_no_longer_in_scene = {}
        self.object_start_frame = {}
        self.object_end_frame = {}
        self.track_frame_length = {} #maybe unnecessary
        self.frame_count = 1;
        self.vehicle_type = {};
        self.vehicle_colour = {};
        self.sent_keys = {};

        colour_names = webcolors.names(webcolors.CSS3)
        colour_codes = []

        for colour in colour_names:
            colour_codes.append(webcolors.name_to_hex(colour))
        self.__colour_dict = list(zip(colour_names, colour_codes)) 

        with open(config_path , 'r') as f:
            self.config =yaml.safe_load(f)['yolov5_deepsort']['main']

        # Add the src directory to the module search path
        sys.path.append(os.path.abspath('../src')) #should point here
        
        #Load the identification model from https://github.com/hoanhle/Vehicle-Type-Detection
        
        # Get YOLO Model Parameter
        self.YOLO_MODEL_NAME = self.config['model_name']
        
        print("test")
        
        # Visualization Parameters
        self.DISP_FPS = self.config['disp_fps'] 
        self.DISP_OBJ_COUNT = self.config['disp_obj_count']
        
        self.object_detector = YOLOv5Detector(model_name=self.YOLO_MODEL_NAME)
        self.tracker = DeepSortTracker()

    def run_model(self):
        while cap.isOpened():
    
            success, img = cap.read() # Read the image frame from data source 
         
            start_time = time.perf_counter()    #Start Timer - needed to calculate FPS
            
            # Object Detection
            results = self.object_detector.run_yolo(img)  # run the yolo v5 object detector 
            
            #TODO: Maybe put in here a check to see if an object is new and to start counting its frames
        
            detections , num_objects= self.object_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[1]) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected
        
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
        
            tracks_current = self.tracker.object_tracker.update_tracks(detections, frame=img)
            self.tracker.display_track(self.track_history , tracks_current , img)
        
            #TODO: get the subimage defined by the bounding boxes from the tracker/detector
            # Pass the subimage to the vehicle classifier model and the average colour subroutine - ONCE
        
            to_be_destroyed = []
        
            if(self.frame_count % 2 == 0):
            
                for key in self.track_history:
                    if(not any(key == value.track_id for value in tracks_current)): #if the key has left the scene
                        to_be_destroyed.append(key)
                    elif key not in self.object_start_frame:
                        self.object_start_frame[key] = self.frame_count
                    elif((key in self.object_end_frame) & (key not in self.vehicle_colour)):
                        to_be_destroyed.append(key)
            
                for key in to_be_destroyed: #deal with the tracks which have left the scene
                    self.objects_no_longer_in_scene[key] = self.track_history.get(key, [])
                    del self.track_history[key]
                    self.object_end_frame[key] = self.frame_count
                    if((key in self.object_end_frame) & (key not in self.vehicle_colour)):
                        del self.object_start_frame[key]
                        del self.object_end_frame[key]
                        del self.objects_no_longer_in_scene[key]
                        # it shouldn't be in vehicle_colour or vehicle_type
        
                #print(type(tracks_current[0].track_id))
                
                #TODO: get the subimage defined by the bounding boxes from the tracker/detector
                # Pass the subimage to the vehicle classifier model and the average colour subroutine - ONCE
            
                for key in self.track_history: #should have got rid of the ones not in the scene
                    if(key not in self.vehicle_colour):
                        if((self.frame_count - self.object_start_frame[key] > 4) & (key not in self.object_end_frame)):
                            print("detecting vehicle type for " + str(key));
                            vehicle_colour_local, subimage = get_colour_from_subimage(key, tracks_current, img, self.__colour_dict) 
                            if(vehicle_colour_local == "AGAIN"):
                                continue
                            self.vehicle_colour[key] = vehicle_colour_local
                            self.vehicle_type[key] = identify_vehicle_type(subimage, self.__identification_model, self.__identification_dictionary) #TODO: identifier is not the most accurate, could need further training
            
            #DEBUG CODE BELOW HERE
            for key in self.objects_no_longer_in_scene:
                print("Car ID " + key + " entered at frame: " + str(self.object_start_frame[key]) + " and left at frame: " +  str(self.object_end_frame[key]) + " with colour: " + str(self.vehicle_colour[key]) + " and type: " + str(self.vehicle_type[key]))

            print("THE CARS CURRENTLY IN THE SCENE ARE: " + str([track.track_id for track in tracks_current]))
            #print(self.objects_no_longer_in_scene)
            #END
        
            
            # FPS Calculation
            end_time = time.perf_counter()
            total_time = end_time - start_time
            fps = 1 / total_time
        
        
            # Descriptions on the output visualization
            cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(img, f'MODEL: {self.YOLO_MODEL_NAME}', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(img, f'TRACKED CLASS: {self.object_detector.tracked_class}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(img, f'TRACKER: {self.tracker.algo_name}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(img, f'DETECTED OBJECTS: {num_objects}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            cv2.imshow('img',img)
        
            
        
        
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
            self.frame_count += 1
        
            
        # Release and destroy all windows before termination
        cap.release()

    def delete_sent_items(self):
        for key in self.sent_keys:
            del self.objects_no_longer_in_scene[key]
            del self.vehicle_colour[key]
            del self.vehicle_type[key]
            del self.object_start_frame[key]
            del self.object_end_frame[key]
            del self.sent_keys[key]
        

    #boring getters and setters
        
    def get_vehicle_type(self):
        return self.vehicle_type

    def get_vehicle_colour(self):
        return self.vehicle_colour

    def get_track_frame_length(self):
        return self.track_frame_length

    def get_track_history(self):
        return self.track_history

    def get_object_start_frame(self):
        return self.object_start_frame

    def get_object_end_frame(self):
        return self.object_end_frame

    def get_objects_no_longer_in_scene(self):
        return self.objects_no_longer_in_scene

    def set_vehicle_type(self, newval):
        self.vehicle_type = newval

    def set_vehicle_colour(self, newval):
        self.vehicle_colour = newval

    def set_track_frame_length(self, newval):
        self.track_frame_length = newval

    def set_track_history(self, newval):
        self.track_history = newval

    def set_object_start_frame(self, newval):
        self.object_start_frame = newval

    def set_object_end_frame(self, newval):
        self.object_end_frame = newval

    def set_objects_no_longer_in_scene(self, newval):
        self.objects_no_longer_in_scene = newval
