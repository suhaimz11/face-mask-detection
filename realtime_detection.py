import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained mask detection model
model = load_model('mask_detector_model.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the font for text on the frame
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop the face region from the frame
        face_region = frame[y:y+h, x:x+w]
        
        # Resize the face region to the model's expected input size (150x150)
        resized_face = cv2.resize(face_region, (150, 150))
        
        # Convert the resized face image to an array and normalize
        img_array = image.img_to_array(resized_face)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize
        
        # Get model prediction
        prediction = model.predict(img_array)
        
        # Print raw prediction values for debugging
        print(f"Prediction raw output: {prediction}")
        
        # Use a higher threshold (e.g., 0.7) for better classification
        label = "Mask" if prediction[0] > 0.7 else "No Mask"
        
        # Set label color based on prediction
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # Green for "Mask", Red for "No Mask"
        
        # Put the label text on the frame
        cv2.putText(frame, label, (x, y-10), font, 1, color, 2, cv2.LINE_AA)
    
    # Show the frame with detected faces and mask prediction
    cv2.imshow("Mask Detection", frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
