# Object Tracking with YOLO and DeepOCSORT

## Introduction

This script combines YOLO (You Only Look Once) for object detection and DeepOCSORT (Online Object Tracking with Siamese Networks) for object tracking. It uses Ultralytics YOLO and the DeepOCSORT tracker to perform real-time object tracking.

### Clone the Repository

```bash
git clone https://github.com/asimthaha/Yolo_v8_models.git
cd Yolo_v8_models
```
### Creating and activating a virtual environment (optional)
```
python -m venv env_name
env_name\Scripts\activate
```

### Install Dependencies
```
pip install -r requirements.txt
```

<details>
<summary>Object Counting using YOLOv8 and Ultralytics</summary>


This repository demonstrates object counting using YOLOv8 and Ultralytics.

### Run the Object Counting Script
```
python object_counting.py
```
The YOLOv8 weights file (yolov8n.pt) from the Ultralytics repository will be downloaded into the models directory.

### Paramters
This script uses YOLOv8 for object detection and Ultralytics for object counting. The object_counting.py script captures a video, applies object tracking, and counts the number of specified classes (e.g., person) that cross a vertical line in the video.

- video_path: Path to the input video file (0 for webcam view).
- line_points: Coordinates of the vertical line in the format [(x1, y1), (x2, y2)].
- classes_to_count: List of classes to count (e.g., [0] for person).
- draw_tracks: Set to True to visualize object tracks.

### Output
The script generates an output video (obj_count.avi) with object counting annotations.
Feel free to customize the parameters and experiment with different videos.


https://github.com/asimthaha/Yolo_v8_models/assets/88647020/946c1458-f81c-4de4-bb4e-107caee62d02



</details>

<details>
<summary>Object Tracking and Reidentification using YOLOv8, DeepSORT, Reid Models</summary>

### Run the Object Tracking Script
```
python object_tracking.py
```
The weights file needed will be downloaded into the models directory automatically.

<details>
<summary>Select Yolo model</summary>

+ yolov8n       # bboxes only
+ yolo_nas_s    # bboxes only
+ yolox_n       # bboxes only
+ yolov8n-seg   # bboxes + segmentation masks
+ yolov8n-pose  # bboxes + pose estimation
</details>

<details>
<summary>Select ReID model</summary>

+ lmbn_n_cuhk03_d.pt               # lightweight
+ osnet_x0_25_market1501.pt
+ mobilenetv2_x1_4_msmt17.engine
+ resnet50_msmt17.onnx
+ osnet_x1_0_msmt17.pt
+ clip_market1501.pt               # heavy
+ clip_vehicleid.pt
</details>

### Paramters
This script uses YOLOv8 for object detection and DeepSORT for object counting. The object_tracking.py script captures a video, applies object tracking, and reidentifies the class

- video_path: Path to the input video file (0 for webcam view).

### Output
The script generates an output video with object tracking and reidentification annotations. Though the model does not provide an accurate solution on black and white videos (that i used) it works well on live camera streams.
Feel free to customize the parameters and experiment with different videos.


https://github.com/asimthaha/Yolo_v8_models/assets/88647020/5c3cc79d-f2ac-4f1a-b876-31bda5c97f80


</details>

## References

+ https://github.com/mikel-brostrom/yolo_tracking.git
+ https://github.com/djidje/deep-person-reid-1.git
+ https://github.com/KaiyangZhou/deep-person-reid.git
