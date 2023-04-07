import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Define the paths to your dataset
dataset_path = "forest_images_dataset"
fire_path = os.path.join(dataset_path, "fire")
non_fire_path = os.path.join(dataset_path, "nonfire")

# Load the images and extract features using OpenCV
def extract_features(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (32, 32))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((32, 32), (16,16), (8,8), (8,8), 9)
    features = hog.compute(gray)
    return features.flatten()


fire_images = [os.path.join(fire_path, f) for f in os.listdir(fire_path)]
non_fire_images = [os.path.join(non_fire_path, f) for f in os.listdir(non_fire_path)]

X = []
y = []

for image_path in fire_images:
    features = extract_features(image_path)
    X.append(features)
    y.append(1)

for image_path in non_fire_images:
    features = extract_features(image_path)
    X.append(features)
    y.append(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a support vector machine (SVM) classifier
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing set
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the trained model to a file
with open("fire_classification_model_version_1.pkl", "wb") as f:
    pickle.dump(clf, f)
