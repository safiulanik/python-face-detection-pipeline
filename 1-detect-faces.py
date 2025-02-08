"""
Step 1: Face recognition

Detect the faces from the [test-image](test-image.png) picture and
store them under the `stored-faces` folder
"""
from typing import List

# importing the cv2 library
import cv2

# loading the haar case algorithm file into alg variable
alg = "haarcascade_frontalface_default.xml"
# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(alg)
# loading the image path into file_name variable - replace <INSERT YOUR IMAGE NAME HERE> with the path to your image
# file_name = "entire_flow.jpg"
file_name = "cefalo_cricket_nerdy_ninjas.jpg"
# read the image as grayscale
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
# creating a black and white version of the image
# gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# detecting the faces
faces = haar_cascade.detectMultiScale(
    img, scaleFactor=1.05, minNeighbors=50, minSize=(100, 100)
)

i = 12
# for each face detected
for x, y, w, h in faces:
    # crop the image to select only the face
    cropped_image = img[y: y + h, x : x + w]
    # loading the target image path into target_file_name variable  -
    # replace <INSERT YOUR TARGET IMAGE NAME HERE> with the path to your target image
    target_file_name = '22/' + str(i) + '.jpg'
    cv2.imwrite(
        target_file_name,
        cropped_image,
    )
    i = i + 1
print(f"Detected {len(faces)} faces")
