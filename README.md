# YOLOv11 Model Training and Inference

This project demonstrates how to train and use a **YOLOv11** model for object detection and classification. The model is specifically fine-tuned to identify and categorize four types of microorganisms: `Cylindrotheca`, `Rhizosolenia`, `Ciliate_mix`, and `Leptocylindrus`.

## Overview

The project workflow consists of the following key steps:

1.  **Environment Setup**: Installing the necessary libraries.
2.  **Dataset Configuration**: Preparing a custom dataset using a YAML file.
3.  **Model Training**: Fine-tuning a pre-trained YOLOv11n model on the custom dataset.
4.  **Inference**: Using the trained model to perform real-time object detection on a video file.

-----

## Setup

### Prerequisites

  - Python 3.8+
  - A system with a CPU (the provided training logs indicate CPU usage)
  - The required Python packages listed in `requirements.txt`.

### Installation

You can install all the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

[cite\_start]The `requirements.txt` file specifies `ultralytics` and `cv`, which handle the core model operations and computer vision tasks, respectively[cite: 754].

-----

## Dataset Configuration

The dataset for this project is configured in the `file.yaml` file, which defines the paths and class information for the model.

**`file.yaml` content:**

```yaml
# data.yaml for YOLOv8 segmentation
path: datasets  # base dataset directory
train: images/train
val: images/val

# number of classes
nc: 4

# class names
names:
  0: Cylindrotheca
  1: Rhizosolenia
  2: Ciliate_mix
  3: Leptocylindrus
```

This configuration tells the model where to find the training and validation images, specifies that there are `4` classes, and provides the names for each class.

-----

## Model Training

The training process uses a pre-trained `yolo11n.pt` model and adapts it to the custom dataset. The training script is found in `model_training.ipynb`.

**Training parameters:**

  - [cite\_start]**Model**: A YOLOv11n model was built from `yolo11n.yaml` and loaded with pre-trained weights from `yolo11n.pt`[cite: 57].
  - [cite\_start]**Epochs**: The model was trained for **20 epochs**[cite: 57].
  - [cite\_start]**Image Size**: Images were scaled to **640x640** pixels[cite: 57].
  - [cite\_start]**Optimizer**: The optimizer was automatically selected as `AdamW` with a learning rate of `0.00125` and a momentum of `0.9`[cite: 57].

[cite\_start]The training took approximately **20.713 hours** to complete, and the best-performing weights were saved to `C:\Users\ASUS\runs\detect\train\weights\best.pt`[cite: 57].

### Training Performance

The final performance metrics on the validation dataset after 20 epochs are as follows:

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| all | 0.881 | 0.903 | 0.933 | 0.613 |
| Cylindrotheca | 0.644 | 0.892 | 0.845 | 0.396 |
| Rhizosolenia | 0.961 | 0.958 | 0.986 | 0.736 |
| Ciliate\_mix | 0.993 | 0.994 | 0.989 | 0.772 |
| Leptocylindrus | 0.924 | 0.768 | 0.914 | 0.548 |

[cite\_start]The model achieved an overall `mAP50-95` of **0.613**[cite: 57].

-----

## Usage

### Real-Time Detection

The `model_training.ipynb` notebook also includes a code snippet for performing real-time object detection on a video file using the trained model.

```python
import cv2
from ultralytics import YOLO

# Load the best trained model
model = YOLO("C:/Users/ASUS/runs/detect/train/weights/best.pt")

# Load your video file
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the frame
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv11 Detection", annotated_frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```
