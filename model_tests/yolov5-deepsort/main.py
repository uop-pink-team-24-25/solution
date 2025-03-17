import sys
import cv2
import time
import yaml
import numpy as np
import random

from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QGridLayout, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import pyqtgraph as pg  # For plotting

from src.detector import YOLOv5Detector
from src.tracker import DeepSortTracker
from src.dataloader import cap  # Video source

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


class ProcessingThread(QThread):
    """
    Runs YOLOv5 & DeepSORT in a background thread.
    """
    detection_complete = pyqtSignal(np.ndarray, int)  # Emit processed frame & FPS
    
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

            img = self.latest_frame.copy()  # Copy the latest frame
            self.latest_frame = None  # Prevent reprocessing the same frame

            start_time = time.perf_counter()

            # Object Detection & Tracking
            results = object_detector.run_yolo(img)
            detections, num_objects = object_detector.extract_detections(
                results, img, height=img.shape[0], width=img.shape[1]
            )
            tracks_current = tracker.object_tracker.update_tracks(detections, frame=img)
            tracker.display_track({}, tracks_current, img)  # Draw bounding boxes

            # FPS Calculation
            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)

            # Emit processed frame (with bounding boxes) and FPS
            self.detection_complete.emit(img, int(fps))

    def stop(self):
        """Stops the thread."""
        self._run_flag = False
        self.quit()
        self.wait()


class VideoApp(QWidget):
    """
    Main application window for displaying video, graph, and buttons.
    """
    
    def __init__(self):
        
        super().__init__()
        
        self.setWindowTitle("Vehicle Detection & Tracking")
        self.setGeometry(100, 100, 900, 700)  # Adjust window size


        # Grid Layout
        self.layout = QGridLayout()

        # QLabel for video
        # QLabel for video (with max size)
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setScaledContents(True)
        self.video_label.setMaximumSize(VIDEO_MAX_WIDTH, VIDEO_MAX_HEIGHT) 
        self.layout.addWidget(self.video_label, 0, 0, 1, 2)  # Video spans two columns

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
        self.dummy1_button = QPushButton("test1", self)
        self.dummy1_button.setFixedWidth(100)
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

    def update_frame(self):
        """Sends the latest frame to processing while keeping UI smooth."""
        global cap
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
            return

        # Send frame to background processing
        if self.processing_thread.latest_frame is None:
            self.processing_thread.latest_frame = img.copy()

        # **DO NOT DISPLAY FRAME IMMEDIATELY** - Wait for processed frame
        if self.latest_displayed_frame is not None:
            img = self.latest_displayed_frame

        # Convert processed frame for display
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Update QLabel with the processed image (with bounding boxes)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
        
        def resizeEvent(self, event):
            """Resize video dynamically while keeping it within limits"""
            if self.latest_displayed_frame is not None:
                self.update_video_display(self.latest_displayed_frame)
            self.video_label.adjustSize()  # 🔹 Ensure QLabel resizes properly
            event.accept()
        
    def update_video_display(self, img):
        """Updates the QLabel with the correctly scaled video frame."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale video to fit QLabel properly
        max_width = min(self.video_label.width(), VIDEO_MAX_WIDTH)  # Max width 800px
        max_height = min(self.video_label.height(), VIDEO_MAX_HEIGHT)  # Max height 450px

        scaled_qpixmap = QPixmap.fromImage(qimg).scaled(max_width, max_height,
                                                        Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation)

        self.video_label.setPixmap(scaled_qpixmap)



    def processed_frame_received(self, img, fps):
        """Receives processed frame (with bounding boxes) and updates UI."""
        self.latest_displayed_frame = img  # Store the processed frame for the next update
        self.fps_label.setText(f"FPS: {fps}")

        # Convert frame to display format
        self.update_video_display(img)

        # 🔹 Simulated Test Data for Graph 🔹
         # Example: 5 vehicle categories (Cars, Trucks, Bikes, etc.)

        # 🔹 Or, if you have real detection data, use detections:
        # test_vehicle_counts = [len(detections)] * 5  # Example: All bars show total detections
        test_vehicle_counts = [2, 5, 7, 3, 9] 
        for i in range(len(test_vehicle_counts)):  # Modify list correctly
            num = random.randint(0, 2)
            if num == 1:
                test_vehicle_counts[i] += 1
            elif num == 2 and test_vehicle_counts[i] > 0:
                test_vehicle_counts[i] -= 1
        self.update_graph(test_vehicle_counts)  # Update the graph
        
        self.update_pie_chart(test_vehicle_counts)
        self.graph_widget.repaint()  # Force UI refresh
        self.pie_chart_view.viewport().update()  # Force pie chart to redraw
        # Convert frame to display format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Update QLabel with the processed image (with bounding boxes)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
    
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
        cap.release()  
        self.processing_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec())
