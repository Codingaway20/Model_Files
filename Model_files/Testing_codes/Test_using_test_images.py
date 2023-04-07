

import cv2
import os
import numpy as np
import pickle

# Define the path to the folder containing the images to classify
classify_path = "fire_test_images"

# Load the trained model from the file
with open("fire_classification_model.pkl", "rb") as f:
    clf = pickle.load(f)

# Load the images to classify and extract features using OpenCV
def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    features = hog.compute(gray)
    return features.flatten()[:324]

classify_images = [os.path.join(classify_path, f) for f in os.listdir(classify_path)]

for image_path in classify_images:
    features = extract_features(image_path)
    label = clf.predict([features])[0]
    if label == 1:
        print(image_path, "is a fire")
    else:
        print(image_path, "is not a fire")
