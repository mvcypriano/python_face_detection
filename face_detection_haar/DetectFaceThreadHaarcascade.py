# This function creates a new thread every 'time_sec_' seconds if a thread
# executing the program is not already running. This thread checks for a face
# and if the face is persisted for more than 'face_appearance_max_' times a
# picture is taken and saved as face_img.jpg
from imutils.video import VideoStream
from RepeatedTimer import RepeatedTimer
import numpy as np
import imutils
import time
import cv2
import threading
import sys

# Global parameters
confidence_ = 0.6
face_size_ = 200
time_sec_ = 0.5
face_appearance_ = [0];


# function that checks for face and save it.
def detect_face(vs):
    frame = vs.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # detect a person face for more than 2 seconds before saving image
    if len(faces) == 1:
        print("Detectou")
        face_appearance_[0] = face_appearance_[0] + 1
        if face_appearance_[0] >= 4:
            application.stop()
            face_appearance_[0] = 0
            print("CRIOU IMAGEM")
            cv2.imwrite('face_img.jpg', frame)
            # Read the binary file to send data to API
            #body = ""
            #filename = 'face_img.jpg'
            #f = open(filename, "rb")
            #body = f.read()
            #f.close()
            application.start()
    else:
        print("Detectou nenhum ou mais de 1")
        face_appearance_[0] = 0


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# initialize the video stream
# change src=0 to the camera source input
vs = VideoStream(src=0).start()
time.sleep(1.0)
print("Video capture started")

# Calls detect_face every time_sec_ seconds
application = RepeatedTimer(time_sec_, detect_face, vs)
