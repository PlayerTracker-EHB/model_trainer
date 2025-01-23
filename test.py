import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
MODEL_PATH = './trained_model.pt'  # Update with actual path
model = YOLO(MODEL_PATH)

# Video source
VIDEO_PATH = './video.mp4'  # Update with actual path

# Define color ranges in HSV format
COLOR_RANGES = {
    "blue": ((100, 150, 50), (140, 255, 255)),
    "red": ((0, 150, 50), (10, 255, 255)),
    "red_alt": ((170, 150, 50), (180, 255, 255)),  # Red wraps around in HSV
    "yellow": ((20, 150, 50), (30, 255, 255)),
    "green": ((40, 150, 50), (70, 255, 255)),
    "white": ((0, 0, 200), (180, 50, 255)),
}

def classify_color(bbox_image):
    """Classify the shirt color of a player using HSV color detection."""
    hsv_image = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2HSV)
    for color, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 0:
            return color
    return "unknown"

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError("Error: Could not open video.")

team_colors = defaultdict(set)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)  # Run YOLO inference
    
    for result in results:
        for bbox in result.boxes.data:
            x1, y1, x2, y2, confidence, cls = map(int, bbox[:6])
            bbox_image = frame[y1:y2, x1:x2]  # Extract bounding box region
            
            shirt_color = classify_color(bbox_image)
            if shirt_color != "unknown":
                player_id = f"{shirt_color}_{x1}_{y1}"
                team_colors[shirt_color].add(player_id)

cap.release()

# Display player count per team
print("Player count per team:")
for color, players in team_colors.items():
    print(f"Team {color}: {len(players)} players")
