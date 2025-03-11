import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
from imgbeddings import imgbeddings
from typing import List

# Directory structure
DATA_DIR = "data"
STORED_FACES_DIR = os.path.join(DATA_DIR, "stored_faces")
TEST_IMAGE = "test-image.png"
HAAR_CASCADE_XML = "haarcascade_frontalface_default.xml"

os.makedirs(STORED_FACES_DIR, exist_ok=True)

# Load face detection model
haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_XML)
ibed = imgbeddings()


def detect_faces(image_path: str) -> List[np.ndarray]:
    """Detects faces in an image and returns cropped face images."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = haar_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=50, minSize=(100, 100))
    cropped_faces = [img[y:y+h, x:x+w] for x, y, w, h in faces]
    return cropped_faces


def store_faces(faces: List[np.ndarray]):
    """Stores detected faces in the designated directory."""
    for i, face in enumerate(faces):
        face_path = os.path.join(STORED_FACES_DIR, f"face_{i}.jpg")
        cv2.imwrite(face_path, face)


def calculate_embeddings(image_list: List[np.ndarray]) -> List[np.ndarray]:
    """Calculates embeddings for a list of images."""
    return [ibed.to_embeddings(Image.fromarray(img))[0] for img in image_list]


def find_similar_faces(target_embedding: np.ndarray, stored_embeddings: List[np.ndarray], top_n: int = 3):
    """Finds top-N most similar face embeddings using cosine similarity."""
    similarities = [np.dot(target_embedding, emb) / (np.linalg.norm(target_embedding) * np.linalg.norm(emb)) for emb in stored_embeddings]
    top_indices = np.argsort(similarities)[-top_n:][::-1]  # Sort in descending order
    return [stored_embeddings[i] for i in top_indices], top_indices


def open_image(path):
    image_viewer_from_command_line = {
        'linux': 'xdg-open',
        'win32': 'explorer',
        'darwin': 'open'
    }[sys.platform]
    subprocess.run([image_viewer_from_command_line, path])


# Step 1: Detect and store faces
faces = detect_faces(TEST_IMAGE)
store_faces(faces)
print(f"Stored {len(faces)} faces.")

# Step 2: Calculate embeddings for stored faces
stored_face_images = [cv2.imread(os.path.join(STORED_FACES_DIR, f"face_{i}.jpg"), cv2.IMREAD_GRAYSCALE) for i in range(len(faces))]
stored_embeddings = calculate_embeddings(stored_face_images)

# Step 3: Find similar faces in-memory
target_image_path = "solo-image.png"
target_faces = detect_faces(target_image_path)
if target_faces:
    target_embedding = calculate_embeddings([target_faces[0]])[0]
    similar_faces, indices = find_similar_faces(target_embedding, stored_embeddings, top_n=1)
    print("Found similar face(s).")
    for i in indices:
        image_path = os.path.join(STORED_FACES_DIR, f"face_{i}.jpg")
        open_image(image_path)
else:
    print("No face detected in target image.")
