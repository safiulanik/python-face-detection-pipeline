# Face Detection Pipeline

## Overview
This project is a face detection and recognition pipeline using OpenCV and imgbeddings. It detects faces in an image, extracts their embeddings, and finds similar faces in-memory without using a database.

## Folder Structure
face-detection-pipeline/ 

│── data/

│ &nbsp; &nbsp; &nbsp; │── stored_faces/ # Folder where detected faces are stored 

│── haarcascade_frontalface_default.xml # Haar cascade model for face detection 

│── test-image.png # Input image for detection 

│── solo-image.png # Image used for similarity comparison 

│── face_detection.py # Main script for the face detection pipeline 

│── README.md # Documentation


## Installation

Ensure you have Python 3 installed. Then, install the required dependencies:
```
pip install opencv-python numpy Pillow imgbeddings
```

## How It Works

1. <b>Face Detection</b>: Detects faces in test-image.png using OpenCV’s Haar cascade classifier.

2. <b>Face Storage</b>: Extracted faces are stored in the data/stored_faces/ directory.

3. <b>Embeddings Calculation</b>: Each detected face is converted into a numerical embedding using imgbeddings.

4. <b>Similarity Matching</b>: A target face from solo-image.png is compared to stored embeddings using cosine similarity.

## Running the Pipeline

Run the script to execute the face detection pipeline:
```
python face_detection.py
```
It will:

- Detect and store faces from test-image.png

- Compute embeddings for the detected faces

- Compare a face from solo-image.png against stored faces and find similar matches

## Important Concepts

### Haar Cascade Classifier

A pre-trained model used for face detection. It scans an image to find face patterns based on edge detection and feature matching.

### Image Embeddings

A numerical representation of an image that captures its features. This allows for similarity comparison between images.

### Cosine Similarity

A metric used to compare two embeddings by measuring the cosine of the angle between them. A value close to 1 indicates high similarity.

## Future Improvements

- Replace Haar cascade with a deep learning-based face detector (e.g., MTCNN or Dlib).

- Use a more robust similarity search like FAISS.

- Support for multiple test images and batch processing.

## License

This project is open-source under the MIT License.