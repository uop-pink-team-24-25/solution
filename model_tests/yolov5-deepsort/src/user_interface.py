import sys
import cv2
import time
import yaml
import numpy as np
import random

from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QGridLayout, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import pyqtgraph as pg  # For plotting

from detector import YOLOv5Detector
from tracker import DeepSortTracker
from dataloader import cap  # Video source
import webcolors

from detector import YOLOv5Detector
from tracker import DeepSortTracker
from dataloader import cap
from colour_getter import get_colour_from_subimage

colour_names = webcolors.names(webcolors.CSS3)
colour_codes = []
for colour in colour_names:
    colour_codes.append(webcolors.name_to_hex(colour))

colour_dict = list(zip(colour_names, colour_codes)) 
#needed for the colour getting. This is declared here because python doesn't have constants so it's passed in

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsTextItem
from PyQt6.QtGui import QColor, QBrush

# Load configuration
with open('model_tests/yolov5-deepsort/config.yml', 'r') as f:
    config = yaml.safe_load(f)['yolov5_deepsort']['main']

YOLO_MODEL_NAME = config['model_name']
DISP_FPS = config['disp_fps']
DISP_OBJ_COUNT = config['disp_obj_count']

object_detector = YOLOv5Detector(model_name=YOLO_MODEL_NAME)
tracker = DeepSortTracker()

VIDEO_MAX_WIDTH = 900   # Change this value for different video widths
VIDEO_MAX_HEIGHT = 800  # Change this value for different video heights

objects_no_longer_in_scene = {}

object_start_frame = {}

object_end_frame = {}

track_frame_length = {}

frame_count = 1

vehicle_type = {}

vehicle_colour = {}
data_source = ['./data/cam_1_1.mp4', './data/cam_1_2.mp4', './data/cam_1_3.mp4', './data/cam_2_1.mp4', './data/cam_2_2.mp4', './data/cam_2_3.mp4']

#while cap.isOpened():


def temp_run_model(img):
    """
    Simulates running the model and returns mock results:
    - img: input frame
    - Returns:
        - processed_img: same frame with mock bounding boxes (or real if using YOLO)
        - fps: calculated or mocked
        - vehicle_counts: list of counts for each category (for graphs)
    """
    start_time = time.perf_counter()

    # --- Optional: Real model (if you want to use the real one for now)
    results = object_detector.run_yolo(img)
    detections, num_objects = object_detector.extract_detections(
        results, img, height=img.shape[0], width=img.shape[1]
    )
    tracks_current = tracker.object_tracker.update_tracks(detections, frame=img)
    tracker.display_track({}, tracks_current, img)  # Draw bounding boxes

    # --- Or: Simulate bounding boxes if model is disabled
    # img = cv2.putText(img, "Simulated Model", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    print(detections)
    # FPS
    end_time = time.perf_counter()
    fps = int(1 / (end_time - start_time))

    # Simulated vehicle counts for the graphs
    vehicle_counts = [random.randint(1, 10) for _ in range(5)]

    return img, fps, vehicle_counts

class ProcessingThread(QThread):
    """
    Runs YOLOv5 & DeepSORT in a background thread.
    """
    detection_complete = pyqtSignal(np.ndarray, int, list)  # Emit processed frame & FPS
    
    # 🔹 Adjustable video display size


    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.latest_frame = None  # Store latest frame for processing

    def run(self):
        """ Continuously processes the latest frame in the background. """
        while self._run_flag:
            if self.latest_frame is None:
                continue  # Wait until a new frame is available

            img = self.latest_frame.copy()
            self.latest_frame = None

            processed_img, fps, vehicle_counts = temp_run_model(img)
            self.detection_complete.emit(processed_img, fps, vehicle_counts)  

        def stop(self):
            """Stops the thread."""
            self._run_flag = False
            self.quit()
            self.wait()


class VideoApp(QWidget):
    """
    Main application window for displaying video, graph, and buttons.
    
    # Object Detection
    results = object_detector.run_yolo(img)  # run the yolo v5 object detector 
    
    #TODO: Maybe put in here a check to see if an object is new and to start counting its frames

    detections , num_objects= object_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[1]) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected

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
    
        for key in to_be_destroyed: #deal with the tracks which have left the scene
            objects_no_longer_in_scene[key] = track_history.get(key, [])
            del track_history[key]
            object_end_frame[key] = frame_count
        #print(type(tracks_current[0].track_id))
        
        #TODO: get the subimage defined by the bounding boxes from the tracker/detector
        # Pass the subimage to the vehicle classifier model and the average colour subroutine - ONCE
    
        for key in track_history: #should have got rid of the ones not in the scene
            if(key not in vehicle_type):
               print("detecting vehicle type for " + str(key));
               vehicle_type[key] = "car" #TODO: sort out model, perhaps get the subimage before here and share between the two?
               vehicle_colour[key] = get_colour_from_subimage(key, tracks_current, img, colour_dict) #TODO: FIX DETECTION BOUNDARIES AND DO MEAN
    
        #DEBUG CODE BELOW HERE
        for key in objects_no_longer_in_scene:
            print("Car ID " + key + " entered at frame: " + str(object_start_frame[key]) + " and left at frame: " +  str(object_end_frame[key]) + " with colour: " + str(vehicle_colour[key]))

        print("THE CARS CURRENTLY IN THE SCENE ARE: " + str([track.track_id for track in tracks_current]))
        #END

    """
    def __init__(self):
        self.cap = cv2.VideoCapture(data_source[0])
        self.current_video_index = 0
        super().__init__()
        self.frame_count = 0
        
        self.setWindowTitle("Vehicle Detection & Tracking")
        self.setGeometry(100, 100, 900, 700)  # Adjust window size


        # Grid Layout
        self.layout = QGridLayout()

        # QLabel for video
        # QLabel for video (with max size)
        # Video label (centered, with size limit)

        #self.layout.addWidget(self.video_label, 0, 0, 1, 2)  # Video spans two columns
        # Row 0: Full horizontal layout with prev button, video label, next button
        # QLabel for video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.video_label.setMaximumSize(VIDEO_MAX_WIDTH, VIDEO_MAX_HEIGHT)
        self.video_label.setMinimumSize(200, 150)
        self.video_label.setScaledContents(False)

        # 🟩 Vertical layout to center video vertically
        video_container = QVBoxLayout()
        video_container.addStretch(1)
        video_container.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        video_container.addStretch(1)

        video_wrapper = QWidget()
        video_wrapper.setLayout(video_container)

        # 🟨 Horizontal layout: prev button | video | next button
        video_row = QHBoxLayout()
        self.prev_button = QPushButton("<")
        self.prev_button.setFixedWidth(50)
        self.prev_button.setFixedHeight(50)
        self.prev_button.setStyleSheet("font-size: 18px;")
        self.prev_button.clicked.connect(self.previous_video)
        video_row.addWidget(self.prev_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        video_row.addWidget(video_wrapper, stretch=1)  # Let video expand in center

        self.next_button = QPushButton(">")
        self.next_button.setFixedWidth(50)
        self.next_button.setFixedHeight(50)
        self.next_button.setStyleSheet("font-size: 18px;")
        self.next_button.clicked.connect(self.next_video)
        video_row.addWidget(self.next_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        # ⬇️ Add full row to layout
        self.layout.addLayout(video_row, 0, 0, 1, 3)


        # FPS Display (Above the chart)
        self.fps_label = QLabel("FPS: 0", self)
        self.layout.addWidget(self.fps_label, 1, 0, 1, 2)  # FPS spans across two columns

        # Create a vertical layout for buttons
        button_layout = QVBoxLayout()

        # Pause Button
        self.toggle_button = QPushButton("Pause", self)
        self.toggle_button.setFixedWidth(100)
        self.toggle_button.clicked.connect(self.toggle_video)
        button_layout.addWidget(self.toggle_button)  # Add to vertical layout

        # Dummy1 Button
        self.dummy1_button = QPushButton("change vid", self)
        self.dummy1_button.setFixedWidth(100)
        self.dummy1_button.clicked.connect(self.cycle_video)
        button_layout.addWidget(self.dummy1_button)  # Add to layout

        # Dummy2 Button
        self.dummy2_button = QPushButton("test2", self)
        self.dummy2_button.setFixedWidth(100)
        button_layout.addWidget(self.dummy2_button)  # Add to layout

        # Add vertical layout to the grid
        self.layout.addLayout(button_layout, 2,2)  # Add buttons to the left column

        
        
        # Graph Widget (Centered)
        # Add Bar Chart (Vehicle Count)
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setTitle("Vehicle Count")
        self.graph_widget.setBackground(None)
        self.graph_widget.setYRange(0, 10)
        self.graph_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.graph_widget.setFixedWidth(400)
        self.graph_widget.setFixedHeight(200)
        self.layout.addWidget(self.graph_widget, 2, 0, 1, 1)  # Left side

        # Pie Chart (Using QGraphicsView instead of PyQtGraph)
        self.pie_chart_view = QGraphicsView()
        self.pie_chart_view.setFixedWidth(200)
        self.pie_chart_view.setFixedHeight(200)
        self.pie_chart_view.setStyleSheet("background: transparent;")  # Transparent Background

        # Create a scene for the pie chart
        self.pie_scene = QGraphicsScene()
        self.pie_chart_view.setScene(self.pie_scene)

        # Add Pie Chart to Layout (Right of Bar Chart)
        self.layout.addWidget(self.pie_chart_view, 2, 1, 1, 1)  # Position correctly

        # Apply layout
        self.setLayout(self.layout)

        # Graph Bar Plot
        self.bar_graph = pg.BarGraphItem(x=np.arange(5), height=[0] * 5, width=0.4)
        self.graph_widget.addItem(self.bar_graph)

        # Timer for fast video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

        self.is_paused = False

        # Processing Thread for YOLO (Background)
        self.processing_thread = ProcessingThread()
        self.processing_thread.detection_complete.connect(self.processed_frame_received)
        self.processing_thread.start()

        self.latest_displayed_frame = None  # Store the latest frame to avoid flickering

    def resizeEvent(self, event):
        """Resize video dynamically while keeping it within limits"""
        if self.latest_displayed_frame is not None:
            self.update_video_display(self.latest_displayed_frame)
        self.video_label.adjustSize()  # 🔹 Ensure QLabel resizes properly
        event.accept()
    def cycle_video(self):
        """Cycles to the next video in the data_source list."""
        self.current_video_index = (self.current_video_index + 1) % len(data_source)
        new_video_path = data_source[self.current_video_index]
        print(f"[INFO] Switching to video: {new_video_path}")
        self.change_video(new_video_path)
    def next_video(self):
        """Switch to the next video in the list."""
        self.current_video_index = (self.current_video_index + 1) % len(data_source)
        self.change_video(data_source[self.current_video_index])

    def previous_video(self):
        """Switch to the previous video in the list."""
        self.current_video_index = (self.current_video_index - 1) % len(data_source)
        self.change_video(data_source[self.current_video_index])
    def update_frame(self):
        """Sends the latest frame to processing while keeping UI smooth."""
        #global self.cap
        success, img = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
            return

        # Send frame to background processing
        if self.processing_thread.latest_frame is None:
            self.processing_thread.latest_frame = img.copy()
        """""
        # **DO NOT DISPLAY FRAME IMMEDIATELY** - Wait for processed frame
        if self.latest_displayed_frame is not None:
            img = self.latest_displayed_frame

        # Convert processed frame for display
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Update QLabel with the processed image (with bounding boxes)
        #self.video_label.setPixmap(QPixmap.fromImage(qimg))
        self.frame_count += 1
        """""

        
    def update_video_display(self, img):
        """Updates the QLabel with the correctly scaled video frame."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to fixed max width and height
        scaled_qpixmap = QPixmap.fromImage(qimg).scaled(
            VIDEO_MAX_WIDTH, VIDEO_MAX_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_qpixmap)

    def change_video(self, new_video_path):
        if self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(new_video_path)

    def processed_frame_received(self, img, fps, vehicle_counts):
        """Receives processed frame (with bounding boxes) and updates UI."""
        self.latest_displayed_frame = img  # Store the processed frame for the next update
        self.fps_label.setText(f"FPS: {fps}")

        # Convert frame to display format
        self.update_video_display(img)

        # 🔹 Simulated Test Data for Graph 🔹
         # Example: 5 vehicle categories (Cars, Trucks, Bikes, etc.)

        # 🔹 Or, if you have real detection data, use detections:
        # test_vehicle_counts = [len(detections)] * 5  # Example: All bars show total detections
        self.update_graph(vehicle_counts)
        self.update_pie_chart(vehicle_counts) # Update the graph
        
        # self.update_pie_chart(vehicle_counts)
        self.graph_widget.repaint()  # Force UI refresh
        self.pie_chart_view.viewport().update()  # Force pie chart to redraw
        # Convert frame to display format
        """""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Update QLabel with the processed image (with bounding boxes)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
        """""
    
    def update_graph(self, data):
        """Updates the graph with new test data."""
        self.bar_graph.setOpts(height=data)
        

    def update_pie_chart(self, data):
        """Updates the pie chart with vehicle category counts."""
        self.pie_scene.clear()  # Clear previous slices

        categories = ["Cars", "Trucks", "Motorbikes", "Buses", "Others"]
        total = sum(data)

        if total == 0:
            return  # Avoid division by zero

        start_angle = 0  # Start at 0 degrees
        colors = [QColor("red"), QColor("blue"), QColor("green"), QColor("yellow"), QColor("purple")]

        for i, value in enumerate(data):
            if value == 0:
                continue  # Skip empty categories

            angle = int((value / total) * 360 * 16)  # Convert to QGraphicsScene angle format

            # Create Pie Slice
            pie_slice = QGraphicsEllipseItem(-50, -50, 100, 100)  # Centered
            pie_slice.setStartAngle(start_angle)
            pie_slice.setSpanAngle(angle)
            pie_slice.setBrush(QBrush(colors[i]))  # Set color

            self.pie_scene.addItem(pie_slice)  # Add slice to scene

            # Add Labels **to the right of the pie chart**
            label = QGraphicsTextItem(f"{categories[i]} ({value})")
            label.setPos(60, i * 20 - 40)  # Fixed position on the right
            label.setDefaultTextColor(QColor("white"))  # Make it readable
            self.pie_scene.addItem(label)

            start_angle += angle  # Move to next slice

        self.pie_chart_view.viewport().update()  # Force redraw

    



    def toggle_video(self):
        """Pause or resume the video."""
        if self.is_paused:
            self.is_paused = False
            self.timer.start(30)
            self.toggle_button.setText("Pause")
        else:
            self.is_paused = True
            self.timer.stop()
            self.toggle_button.setText("Resume")

    def closeEvent(self, event):
        """Ensure video capture and threads stop when closing the app."""
        print("[INFO] Closing application...")
        self.cap.release()  
        self.processing_thread.stop()
        event.accept()

    #frame_count += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec())
