import cv2
import numpy as np
import joblib

# Load the trained model
model = joblib.load('fire_classification_model.pkl')


for i in range(10):
    # Define the camera capture
    camera = cv2.VideoCapture(0)

    # Capture a single image
    ret, image = camera.read()

    # Save the captured image to disk
    strr=f'captured_image{i}.jpg'
    cv2.imwrite(strr,image)

    # Preprocess the image for classification
    resized_image = cv2.resize(image, (18, 18))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    flattened_image = gray_image.flatten()

    # Make a prediction using the trained model
    prediction = model.predict([flattened_image])

    # Display the prediction
    if prediction[0] == 0:
        print('The image does not contain fire.')
    else:
        print('The image contains fire!')
    # Release the camera and close the window
    camera.release()
    

cv2.destroyAllWindows()
