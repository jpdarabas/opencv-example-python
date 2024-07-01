import cv2
import sys
import numpy as np

# Loading cascade classifiers
face = cv2.CascadeClassifier('haar-cascade-files/haarcascade_frontalface_alt2.xml')
eyes = cv2.CascadeClassifier('haar-cascade-files/haarcascade_eye_tree_eyeglasses.xml')

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
    print(s)

source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Array to store y coordinates of eyes
y_array = np.zeros(42)
count = 0

# Variable to check if the user is dozing off to freeze the y_array
isDozingOff = False

# Loop to capture frames
while cv2.waitKey(1) != 27: # Escape key to exit
    has_frame, frame = source.read()
    if not has_frame:
        break

    height,width = frame.shape[:2]

    # Converting the frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting face
    face_edges = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    face_height = 0
    for (x, y, w, h) in face_edges:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_height = h
    
    # Detecting eyes (the size is relative to the face size)
    eyes_edges = eyes.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(face_height/5, face_height/5), maxSize=(face_height/3.5, face_height/3.5))
    for (x, y, w, h) in eyes_edges:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Storing the y coordinates of the eyes in the array
        if count == len(y_array):
            if not isDozingOff:
                y_array = np.roll(y_array, -1)
            y_array[-1] = int(y)
        else: 
            y_array[count] = int(y)
            count += 1

    # Detecting if the user is dozing off using the y coordinates of the eyes
    if (y_array[-1] -(face_height*0.5)) > y_array[0]:
        cv2.putText(frame, "You're dozing off", (w-20, h+70), cv2.FONT_HERSHEY_PLAIN, 2.3, (0, 255, 0), 2, cv2.LINE_AA)
        isDozingOff = True
    else:
        isDozingOff = False

    # Displaying the frame
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)