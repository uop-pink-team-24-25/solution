import os
import sys
import time
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from src import detector

base_dir = os.path.dirname(os.path.abspath(__file__)) 
src_dir = os.path.join(base_dir, 'src') 
sys.path.insert(0, src_dir) 

def parse_xml_label(label_path):
   
    boxes = []
    tree = ET.parse(label_path)
    root = tree.getroot()
    
    # Extract image size from XML
    size = root.find('size')
    xml_width = int(size.find('width').text)
    xml_height = int(size.find('height').text)
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name.lower() != 'car': 
            continue
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        boxes.append((2, x1, y1, x2, y2))  # Class ID 2 for cars
    
    return boxes, xml_width, xml_height

def compute_iou(boxA, boxB):
    """ (No changes here) """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def benchmark_model(images_dir, labels_dir, detector, iou_threshold=0.5):
    """ (Mostly unchanged, key modifications below) """

    total_time = 0.0
    total_images = 0
    total_TP = 0  
    total_FP = 0  
    total_FN = 0  

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, filename.replace('.jpg', '.xml')) 
            if not os.path.exists(label_path):
                print(f"Label for {filename} not found, skipping.")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load {filename}, skipping.")
                continue
            
            img_height, img_width = image.shape[:2]
            gt_boxes, xml_width, xml_height = parse_xml_label(label_path)  # Parse XML

            # Scale boxes if image dimensions don't match
            if xml_width != img_width or xml_height != img_height:
                scale_x = img_width / xml_width
                scale_y = img_height / xml_height
                scaled_boxes = []
                for box in gt_boxes:
                    cls, x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    scaled_boxes.append((cls, x1, y1, x2, y2))
                gt_boxes = scaled_boxes

            start_time = time.time()
            labels, bbox_cords = detector.run_yolo(image)
            inference_time = time.time() - start_time
            total_time += inference_time
            total_images += 1


            detections, _ = detector.extract_detections((labels, bbox_cords), image, img_height, img_width)

            print(f"Raw YOLO Output - Labels: {labels}")
            print(f"Raw YOLO Output - Bounding Boxes: {bbox_cords}")


            # Convert detections to (class, x1, y1, x2, y2) format
            pred_boxes = []
            for det in detections:
                bbox, conf, feature = det  # (bbox, confidence, class_name)
                x, y, w, h = bbox  # Extract coordinates from bbox 
                x1, y1, x2, y2 = x, y, x + w, y + h
                det_class_id = 2 if feature == 'car' else -1
                pred_boxes.append((det_class_id, x1, y1, x2, y2))

            print(f"\nImage: {filename}")
            print(f"Ground Truth: {gt_boxes}")  #
            print(f"Predicted Boxes: {pred_boxes}")

            # ======== Compute Precision and Recall =========
            matched_pred = set()
            TP = 0

            for gt in gt_boxes:
                gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt
                match_found = False
                for idx, pred in enumerate(pred_boxes):
                    pred_class, pred_x1, pred_y1, pred_x2, pred_y2 = pred
                    if str(gt_class) != str(pred_class):
                        continue
                    
                    iou = compute_iou((gt_x1, gt_y1, gt_x2, gt_y2), (pred_x1, pred_y1, pred_x2, pred_y2))
                    if iou >= iou_threshold:
                        TP += 1
                        matched_pred.add(idx)
                        match_found = True
                        break  # to limit  detections per ground truth box to 1

                if not match_found:
                    total_FN += 1  # A car was not detected

            total_FP += (len(pred_boxes) - len(matched_pred))  # false positives
            total_TP += TP  # Count true positives

    # FPS
    avg_inference_time = total_time / total_images if total_images else 0
    fps = total_images / total_time if total_time else 0

    # Precision and Recall
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

    metrics = {
        "total_images": total_images,
        "avg_inference_time": avg_inference_time,
        "fps": fps,
        "precision": precision,
        "recall": recall
    }

    return metrics

if __name__ == "__main__":
    base_dir = os.getcwd()
    images_directory = os.path.join(base_dir, "benchmark", "screenshots")
    labels_directory = os.path.join(base_dir, "benchmark", "labelled_screenshots")
    detector = detector.YOLOv5Detector(model_name="yolov5n.pt", config_path=os.path.join(base_dir, "config.yml"))
    results = benchmark_model(images_directory, labels_directory, detector)
    print("\nBenchmarking Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")