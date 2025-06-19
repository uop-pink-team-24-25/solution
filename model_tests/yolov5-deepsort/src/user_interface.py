import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*torch.*")

import sys, time, cv2, numpy as np
from collections import Counter
from PyQt6.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QGridLayout, QSizePolicy, QInputDialog, QMessageBox, QLineEdit,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsTextItem
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QBrush
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import pyqtgraph as pg
import webcolors
import queue

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import BatchNormalization as TF_BatchNormalization

from detector import YOLOv5Detector
from tracker import DeepSortTracker
from colour_getter import get_colour_from_subimage
from type_identifier import identify_vehicle_type
from model_runner import ai_model

import csv
import json
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Constants
VIDEO_MAX_WIDTH = 900
VIDEO_MAX_HEIGHT = 800
ACTUAL_WIDTH = 1920
ACTUAL_HEIGHT = 1080
video_files = ['Data.mp4', 'Data2.mp4']
data_source = [str(PROJECT_ROOT / "data" / f) for f in video_files]
HEATMAP_SIZE = (VIDEO_MAX_HEIGHT, VIDEO_MAX_WIDTH)  # matches display resolution
heatmap_accumulator = np.zeros(HEATMAP_SIZE, dtype=np.uint32)

# Setup color dictionary
colour_names = webcolors.names(webcolors.CSS3)
colour_codes = [webcolors.name_to_hex(name) for name in colour_names]
colour_dict = list(zip(colour_names, colour_codes))

# Patch Keras BatchNormalization
def patched_batchnorm(*args, **kwargs):
    if isinstance(kwargs.get('axis'), list):
        kwargs['axis'] = kwargs['axis'][0]
    return TF_BatchNormalization(*args, **kwargs)

get_custom_objects()['BatchNormalization'] = patched_batchnorm

# Load classification model
identification_model = load_model('./mobilenet2.h5')
identification_dictionary = dict(zip(
    range(17),
    ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar',
     'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']
))

# Initialize AI model
MODEL = ai_model("../config.yml", show=False)
MODEL.heatmap = np.zeros((VIDEO_MAX_HEIGHT, VIDEO_MAX_WIDTH), dtype=np.float32)
MODEL.track_centers = {}
MODEL.track_history = {}
MODEL.frame_count = 0
MODEL.object_start_frame = {}
MODEL.object_end_frame = {}
MODEL.vehicle_colour = {}
MODEL.vehicle_type = {}

def convert_to_builtin_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_to_builtin_type(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_data_to_csv(model, path=str(PROJECT_ROOT / "output.csv")):
    objects = model.get_objects_no_longer_in_scene()
    print(f"[INFO] Saving CSV to: {os.path.abspath(path)}")
    with open(path, "w", newline="") as csvfile:
        fieldnames = ["track_id", "start_frame", "end_frame", "vehicle_type", "vehicle_colour", "track_history"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in model.completed_vehicle_data:
            track_id = row["track_id"]
            track_history = row.get("track_history", [])
            track_history_clean = convert_to_builtin_type(track_history)
            row["track_history"] = json.dumps(track_history_clean)
            writer.writerow(row)

def temp_run_model(img):
    
    start_time = time.perf_counter()
    MODEL.heatmap = (MODEL.heatmap * 0.95)

    results = MODEL.object_detector.run_yolo(img)
    detections, _ = MODEL.object_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[1])
    tracks_current = MODEL.tracker.object_tracker.update_tracks(detections, frame=img)
    print(f"[DEBUG] Number of tracks: {len(tracks_current)}")
    for track in tracks_current:
        key = track.track_id
        age = MODEL.frame_count - MODEL.object_start_frame.get(key, 0)

        if key not in MODEL.vehicle_colour and age > 3:
            colour_result, subimage = get_colour_from_subimage(key, tracks_current, img, colour_dict)
            if colour_result != "AGAIN" and subimage is not None:
                MODEL.vehicle_colour[key] = colour_result
                MODEL.vehicle_type[key] = identify_vehicle_type(subimage, identification_model, identification_dictionary)

    MODEL.tracker.display_track(MODEL.track_centers, tracks_current, img)


    if MODEL.frame_count % 2 == 0:
        active_ids = [t.track_id for t in tracks_current]

        for track in tracks_current:
            if track.track_id not in MODEL.object_start_frame:
                MODEL.object_start_frame[track.track_id] = MODEL.frame_count

        for key in list(MODEL.track_history.keys()):
            if key not in active_ids:
                MODEL.object_end_frame[key] = MODEL.frame_count
                MODEL.objects_no_longer_in_scene[key] = MODEL.track_centers.get(key, [])

                if key in MODEL.vehicle_colour and key in MODEL.vehicle_type:
                    MODEL.completed_vehicle_data.append({
                        "track_id": key,
                        "start_frame": MODEL.object_start_frame.get(key),
                        "end_frame": MODEL.object_end_frame.get(key),
                        "vehicle_type": MODEL.vehicle_type.get(key),
                        "vehicle_colour": MODEL.vehicle_colour.get(key),
                        "track_history": MODEL.track_centers.get(key, [])
                    })

                MODEL.track_history.pop(key, None)
                MODEL.track_centers.pop(key, None)


    for track in tracks_current:
        key = track.track_id
        center = (
            int(track.to_tlwh()[0] + track.to_tlwh()[2] / 2),
            int(track.to_tlwh()[1] + track.to_tlwh()[3] / 2)
        )
        x, y = center

        # Ensure bounds are valid
        if 0 <= x < MODEL.heatmap.shape[1] and 0 <= y < MODEL.heatmap.shape[0]:
            cv2.circle(MODEL.heatmap, (x, y), radius=10, color=1, thickness=-1)
            print(f"[DEBUG] Updated heatmap at ({x}, {y}). Current value: {MODEL.heatmap[y, x]}")

        MODEL.track_centers.setdefault(key, []).append(center)

    MODEL.track_history = {t.track_id: t for t in tracks_current}
    MODEL.frame_count += 1

    category_map = ["Cars", "Trucks", "Motorbikes", "Buses", "Others"]
    category_counts = {k: 0 for k in category_map}

    for vt in MODEL.vehicle_type.values():
        vt_lower = vt.lower()
        if "car" in vt_lower or "taxi" in vt_lower or "limousine" in vt_lower:
            category_counts["Cars"] += 1
        elif "truck" in vt_lower or "caterpillar" in vt_lower or "van" in vt_lower:
            category_counts["Trucks"] += 1
        elif "motor" in vt_lower or "bike" in vt_lower:
            category_counts["Motorbikes"] += 1
        elif "bus" in vt_lower:
            category_counts["Buses"] += 1
        else:
            category_counts["Others"] += 1

    colour_counter = Counter(MODEL.vehicle_colour.values())
    top_colours = colour_counter.most_common(5)
    colour_names_for_graph = [name for name, _ in top_colours]
    colour_counts_for_graph = [count for _, count in top_colours]

    vehicle_counts = [category_counts[c] for c in category_map]
    fps = int(1 / (time.perf_counter() - start_time))

    return img, fps, vehicle_counts, colour_counts_for_graph, colour_names_for_graph

class ProcessingThread(QThread):
    detection_complete = pyqtSignal(np.ndarray, int, list, list, list)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.frame_queue = queue.Queue(maxsize=2)  # small size to limit lag

    def run(self):
        while self._run_flag:
            try:
                img = self.frame_queue.get(timeout=1)
                output = temp_run_model(img)
                self.detection_complete.emit(*output)
            except queue.Empty:
                continue

    def stop(self):
        self._run_flag = False
        self.quit()
        self.wait()

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(data_source[0])
        self.current_video_index = 0
        self.video_unlocked = False
        self.is_paused = False
        self.latest_displayed_frame = None

        self.setWindowTitle("Vehicle Detection & Tracking")
        self.setGeometry(100, 100, 900, 700)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.video_label.setMaximumSize(VIDEO_MAX_WIDTH, VIDEO_MAX_HEIGHT)
        self.video_label.setMinimumSize(200, 150)
        self.video_label.setScaledContents(False)
        self.video_label.hide()

        self.setup_layout()
        self.setup_graphs()
        self.setup_timers()

    def setup_layout(self):
        video_container = QVBoxLayout()
        video_container.addStretch(1)
        video_container.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        video_container.addStretch(1)

        video_wrapper = QWidget()
        video_wrapper.setLayout(video_container)

        video_row = QHBoxLayout()
        self.prev_button = QPushButton("<")
        self.prev_button.clicked.connect(self.previous_video)
        self.next_button = QPushButton(">")
        self.next_button.clicked.connect(self.next_video)
        for btn in (self.prev_button, self.next_button):
            btn.setFixedSize(50, 50)
            btn.setStyleSheet("font-size: 18px;")

        video_row.addWidget(self.prev_button)
        video_row.addWidget(video_wrapper, stretch=1)
        video_row.addWidget(self.next_button)

        self.layout.addLayout(video_row, 0, 0, 1, 3)

        self.fps_label = QLabel("FPS: 0", self)
        self.layout.addWidget(self.fps_label, 1, 0, 1, 2)

        button_layout = QVBoxLayout()
        self.toggle_button = QPushButton("Pause", self)
        self.toggle_button.clicked.connect(self.toggle_video)
        self.unlock_button = QPushButton("Enter Password", self)
        self.unlock_button.clicked.connect(self.prompt_password)

        for btn in (self.toggle_button, self.unlock_button):
            btn.setFixedWidth(100)
            button_layout.addWidget(btn)

        self.layout.addLayout(button_layout, 2, 2)
        
    import ast  # for safely parsing track_history strings

    def show_saved_data_summary(self, path=str(PROJECT_ROOT / "output.csv")):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Summary Error", "CSV file not found.")
            return

        from collections import Counter
        import pandas as pd

        df = pd.read_csv(path)

        # -- Most Common Colour --
        colour_counts = Counter(df["vehicle_colour"].dropna())
        most_common_colour = colour_counts.most_common(1)[0][0] if colour_counts else "N/A"

        # -- Average Time in Frame --
        times = []
        for _, row in df.iterrows():
            try:
                if not np.isnan(row["start_frame"]) and not np.isnan(row["end_frame"]):
                    times.append(int(row["end_frame"]) - int(row["start_frame"]))
            except:
                continue
        avg_time = round(sum(times) / len(times), 2) if times else "N/A"

        # -- Most Common Exit Region --
        import ast
        region_counts = Counter()
        for track_str in df["track_history"]:
            try:
                points = ast.literal_eval(track_str)
                if isinstance(points, list) and len(points) > 0 and isinstance(points[-1], list):
                    x, y = points[-1]
                    
                    # Divide the screen into a 3x3 grid
                    col = "Left" if x < ACTUAL_WIDTH  / 3 else "Right" if x > 2 * ACTUAL_WIDTH / 3 else "Center"
                    row = "Top" if y < ACTUAL_HEIGHT  / 3 else "Bottom" if y > 2 * ACTUAL_HEIGHT / 3 else "Middle"
                    region = f"{row} {col}"
                    region_counts[region] += 1
            except:
                continue

        most_common_region = region_counts.most_common(1)[0][0] if region_counts else "Unknown"
        print(f"[DEBUG] Most common region: {most_common_region}")
        print(f"[DEBUG] All regions: {region_counts}")

        # -- Display Summary --
        summary = (
            f"📊 **Saved Data Summary**\n\n"
            f"• Most Common Colour: {most_common_colour}\n"
            f"• Average Time in Frame: {avg_time} frames\n"
            f"• Most Common Exit Region: {most_common_region}"
        )

        QMessageBox.information(self, "Saved Data Summary", summary)




    def setup_graphs(self):
        self.graph_widget = pg.PlotWidget(title="Vehicle Count")
        self.graph_widget.setBackground(None)
        self.graph_widget.setYRange(0, 10)
        self.graph_widget.setFixedSize(400, 200)
        self.layout.addWidget(self.graph_widget, 2, 0, 1, 1)

        self.pie_chart_view = QGraphicsView()
        self.pie_chart_view.setFixedSize(200, 200)
        self.pie_chart_view.setStyleSheet("background: transparent;")
        self.pie_scene = QGraphicsScene()
        self.pie_chart_view.setScene(self.pie_scene)
        self.layout.addWidget(self.pie_chart_view, 2, 1)
        
        # Heatmap Display (Image View)
        # Heatmap Display using QLabel
        self.heatmap_label = QLabel()
        self.heatmap_label.setFixedSize(300, 200)
        self.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap_label.setStyleSheet("border: 1px solid gray;")
        self.layout.addWidget(self.heatmap_label, 3, 0, 1, 2)

    def setup_timers(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.processing_thread = ProcessingThread()
        self.processing_thread.detection_complete.connect(self.processed_frame_received)
        self.processing_thread.start()

    def resizeEvent(self, event):
        if self.latest_displayed_frame is not None:
            self.update_video_display(self.latest_displayed_frame)
        self.video_label.adjustSize()
        event.accept()

    def prompt_password(self):
        password, ok = QInputDialog.getText(self, "Password Required", "Enter password:", echo=QLineEdit.EchoMode.Password)
        if ok and password == "mySecret123":
            self.video_unlocked = True
            self.video_label.show()
            self.unlock_button.setDisabled(True)
            QMessageBox.information(self, "Access Granted", "Video feed unlocked.")
        elif ok:
            QMessageBox.warning(self, "Access Denied", "Incorrect password.")

    def next_video(self):
        self.current_video_index = (self.current_video_index + 1) % len(data_source)
        self.change_video(data_source[self.current_video_index])

    def previous_video(self):
        self.current_video_index = (self.current_video_index - 1) % len(data_source)
        self.change_video(data_source[self.current_video_index])

    def change_video(self, new_video_path):
        if self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(new_video_path)

    def update_frame(self):
        success, img = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        if not self.processing_thread.frame_queue.full():
            self.processing_thread.frame_queue.put_nowait(img.copy())

    def update_video_display(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qimg).scaled(VIDEO_MAX_WIDTH, VIDEO_MAX_HEIGHT, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def processed_frame_received(self, img, fps, vehicle_counts, colour_counts, colour_labels):
        
        self.latest_displayed_frame = img
        self.fps_label.setText(f"FPS: {fps}")
        if self.video_unlocked:
            self.update_video_display(img)
        self.update_graph(colour_counts, colour_labels)
        self.update_pie_chart(vehicle_counts)
        normalized = cv2.normalize(MODEL.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = normalized.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        h, w, ch = heatmap_rgb.shape
        qimg = QImage(heatmap_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.heatmap_label.width(), self.heatmap_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.heatmap_label.setPixmap(pixmap)

    def update_graph(self, data, labels):
        self.graph_widget.clear()
        x = np.arange(len(data))
        bar = pg.BarGraphItem(x=x, height=data, width=0.6)
        self.graph_widget.addItem(bar)
        axis = self.graph_widget.getAxis('bottom')
        axis.setTicks([[(i, label) for i, label in enumerate(labels)]])

    def update_pie_chart(self, data):
        self.pie_scene.clear()
        categories = ["Cars", "Trucks", "Motorbikes", "Buses", "Others"]
        total = sum(data)
        if total == 0:
            return
        start_angle = 0
        colors = [QColor("red"), QColor("blue"), QColor("green"), QColor("yellow"), QColor("purple")]
        for i, value in enumerate(data):
            if value == 0:
                continue
            angle = int((value / total) * 360 * 16)
            pie_slice = QGraphicsEllipseItem(-50, -50, 100, 100)
            pie_slice.setStartAngle(start_angle)
            pie_slice.setSpanAngle(angle)
            pie_slice.setBrush(QBrush(colors[i]))
            self.pie_scene.addItem(pie_slice)
            label = QGraphicsTextItem(f"{categories[i]} ({value})")
            label.setPos(60, i * 20 - 40)
            label.setDefaultTextColor(QColor("white"))
            self.pie_scene.addItem(label)
            start_angle += angle

    def toggle_video(self):
        if self.is_paused:
            self.timer.start(30)
            self.toggle_button.setText("Pause")
        else:
            self.timer.stop()
            self.toggle_button.setText("Resume")
            print("[INFO] Saving current tracking data to CSV...")
            save_data_to_csv(MODEL)
            print("[INFO] Data saved to output.csv")
            self.show_saved_data_summary()
        self.is_paused = not self.is_paused

    def closeEvent(self, event):
        print("[INFO] Closing application...")
        self.cap.release()
        self.processing_thread.stop()
        event.accept()
        



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec())
