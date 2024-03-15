import torch
import glob
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from glob import glob

# Define the labels and other constants
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the intersection rectangle's coordinates
    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)

    # Calculate area of intersection
    intersection_area = max(0, x6 - x5 + 1) * max(0, y6 - y5 + 1)

    # Calculate area of both boxes
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

IMG_HEIGHT = 720
IMG_WIDTH = 1280
labels = ["bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train", "truck"]
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
TRAIN_PATH = 'C:/Users/bhara/Downloads/Data/train'
VAL_PATH = 'C:/Users/bhara/Downloads/Data/val'
MODEL_PATH = 'C:/Users/bhara/Downloads/Data/best.pt'
STEERING_WHEEL_PATH = 'C:/Users/bhara/Downloads/Data/steering_wheel_image.jpg'

# Load the model
model = torch.hub.load('yolov5', 'custom', path=MODEL_PATH, source='local')

# Load the steering wheel image
wheel = cv2.imread(STEERING_WHEEL_PATH)
wheel = cv2.cvtColor(wheel, cv2.COLOR_BGR2GRAY)
rows, cols = wheel.shape

# Get a sample of validation images
val_images = glob(f'{VAL_PATH}/*.jpg')
n_samples = 10
sample_images = np.random.choice(val_images, size=n_samples)

# Calculate accuracy for each image
accuracies = []

for i in range(n_samples):
    img_path = sample_images[i]
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    label_file = os.path.join(VAL_PATH, f'{img_id}.txt')

    with open(label_file, 'r') as f:
        lines = f.readlines()

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(img)
    detected_objects = results.pandas().xyxy[0]  # Get detected objects in the first frame

    # Initialize accuracy variables
    total_objects = len(lines)
    correctly_detected = 0

    for label in lines:
        splits = label.split()
        category = labels[int(splits[0])]
        x_center = float(splits[1]) * IMG_WIDTH
        y_center = float(splits[2]) * IMG_HEIGHT
        width = float(splits[3]) * IMG_WIDTH
        height = float(splits[4]) * IMG_HEIGHT

        for index, row in detected_objects.iterrows():
            if row['name'] == category:  # 'name' is the column containing object class
                detected_box = row[['xmin', 'ymin', 'xmax', 'ymax']]
                iou = calculate_iou(detected_box, [x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2])
                if iou >= 0.5:  # You can adjust this threshold as needed
                    correctly_detected += 1
                    break

    accuracy = correctly_detected / total_objects
    accuracies.append(accuracy)

    # Display the image with detections
    fig, ax = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
    ax[0].axis('off')
    ax[0].imshow(np.squeeze(results.render()))
    ax[1].axis('off')
    # Display the image with detections
    fig, ax = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
    ax[0].axis('off')
    ax[0].imshow(np.squeeze(results.render()))
    ax[1].axis('off')
    ax[1].imshow(wheel, cmap='gray')  # Use the 'wheel' variable
    plt.show()

    plt.show()

# Print the accuracies for each image
for i, accuracy in enumerate(accuracies):
    print(f'Accuracy for image {i + 1}: {accuracy * 100:.2f}%')
