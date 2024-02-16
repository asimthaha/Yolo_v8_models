# Object Counting using YOLOv8 and Ultralytics

This repository demonstrates object counting using YOLOv8 and Ultralytics.

## Clone the Repository

```bash
git clone https://github.com/asimthaha/Yolo_v8_models.git
cd Yolo_v8_models
```
## Creating and activating a virtual environment (optional)
```
python -m venv env_name
env_name\Scripts\activate
```

## Install Dependencies
```
pip install ultralytics
```

## Run the Object Counting Script
```
python object_counting.py
```
The YOLOv8 weights file (yolov8n.pt) from the Ultralytics repository will be downloaded into your root directory.

## Paramters
This script uses YOLOv8 for object detection and Ultralytics for object counting. The object_counting.py script captures a video, applies object tracking, and counts the number of specified classes (e.g., person) that cross a vertical line in the video.

- video_path: Path to the input video file (0 for webcam view).
- line_points: Coordinates of the vertical line in the format [(x1, y1), (x2, y2)].
- classes_to_count: List of classes to count (e.g., [0] for person).
- draw_tracks: Set to True to visualize object tracks.

## Output
The script generates an output video (obj_count.avi) with object counting annotations.
Feel free to customize the parameters and experiment with different videos.
