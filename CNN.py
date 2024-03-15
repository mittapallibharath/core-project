# Import necessary libraries
import cv2
import numpy as np

# Load the ground truth data and model predictions
# Replace these with your actual data and predictions
ground_truth_boxes = [...]  # List of ground truth bounding boxes in [x, y, width, height] format
predicted_boxes = [...]     # List of predicted bounding boxes in [x, y, width, height] format

# Set a threshold for considering a detection as correct (e.g., Intersection over Union threshold)
iou_threshold = 0.5

# Initialize variables for counting true positives and total ground truth objects
true_positives = 0
total_ground_truth = len(ground_truth_boxes)

# Iterate over predicted boxes and compare with ground truth
for pred_box in predicted_boxes:
    for gt_box in ground_truth_boxes:
        # Calculate the Intersection over Union (IoU)
        x1 = max(pred_box[0], gt_box[0])
        y1 = max(pred_box[1], gt_box[1])
        x2 = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
        y2 = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        pred_box_area = pred_box[2] * pred_box[3]
        gt_box_area = gt_box[2] * gt_box[3]

        iou = intersection_area / (pred_box_area + gt_box_area - intersection_area)

        if iou >= iou_threshold:
            true_positives += 1
            # Remove the matched ground truth box to avoid double counting
            ground_truth_boxes.remove(gt_box)
            break

# Calculate precision and recall
precision = true_positives / len(predicted_boxes)
recall = true_positives / total_ground_truth

# Calculate accuracy as the harmonic mean (F1 score)
accuracy = 2 * (precision * recall) / (precision + recall) * 100

print(f"Accuracy: {accuracy:.2f}%")
