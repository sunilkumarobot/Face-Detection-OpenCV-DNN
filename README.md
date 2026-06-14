# Face Detection using OpenCV DNN
## Project Description

This project implements face detection using OpenCV's Deep Neural Network (DNN) module and a pre-trained SSD ResNet model. The system is capable of detecting human faces in both static images and real-time webcam streams.
The project demonstrates the complete deep learning inference pipeline, including image preprocessing, blob creation, model inference, confidence-based filtering, and visualization of detection results.

## Objectives

* Detect faces in images using a pre-trained deep learning model.
* Perform real-time face detection using a webcam.
* Understand the OpenCV DNN inference workflow.
* Visualize detected faces using bounding boxes.

## Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib

## Deep Learning Model

**Model:** Res10 SSD Face Detector

**Framework:** Caffe

Files used:

* deploy.prototxt
* res10_300x300_ssd_iter_140000.caffemodel

The model is trained to detect human faces and returns bounding box coordinates along with confidence scores for each detection.
## OpenCV DNN Workflow

1. Load the pre-trained model.
2. Read an input image or webcam frame.
3. Convert the image into a blob using `blobFromImage()`.
4. Pass the blob to the neural network using `setInput()`.
5. Run inference using `forward()`.
6. Extract confidence scores and bounding box coordinates.
7. Draw bounding boxes around detected faces.
8. Display the output.

## Project Structure

Face-Detection-OpenCV-DNN/

├── image_face_detection.py

├── webcam_face_detection.py

├── README.md

└── screenshots/

---

## Features

### Image Face Detection

Detects faces from an input image and draws bounding boxes around detected faces.

### Real-Time Webcam Face Detection

Captures frames from a webcam and performs face detection in real time using the same DNN model.

### Confidence Threshold Filtering

Only detections above a specified confidence threshold are displayed.

---

## Results

The model successfully detects human faces in both static images and live webcam feeds.

Example output:

* Face localization using bounding boxes.
* Confidence-based face detection.
* Real-time detection performance.

---

## Installation

Clone the repository:

git clone https://github.com/sunilkumarobot/Face-Detection-OpenCV-DNN.git

Install dependencies:

pip install opencv-python numpy matplotlib

---

## Usage

Run image face detection:

python image_face_detection.py

Run webcam face detection:

python webcam_face_detection.py

---

## Learning Outcomes

Through this project, I gained practical experience with:

* OpenCV DNN module
* Deep learning inference
* Blob creation and preprocessing
* Face detection using SSD models
* Real-time computer vision applications
* Bounding box generation and visualization

---

## Future Improvements

* Face blurring for privacy protection
* Face pixelation
* Face recognition
* Mask detection
* Age and gender estimation
* Multi-face tracking in video streams
