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

# Global parameters
confidence_ = 0.6
face_size_ = 200
time_sec_ = 0.2
face_appearance_ = [0];


# function that checks for face and save it.
def detect_face(vs):
    relevant_face_counter = 0
    frame = vs.read()
    # frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the neural network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # check for detections
    # loop through all detections
    for i in range(0, detections.shape[2]):
        # filter detections according to a confidence metric
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_:
            continue

        # certify that a face has at least a certain size, other wise might
        # detect an unwanted person by accident
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        if (endX - startX) < face_size_ or (endY - startY) < face_size_:
            continue

        relevant_face_counter = relevant_face_counter + 1

    # detect a person face for more than 2 seconds before saving image
    if relevant_face_counter == 1:
        print("Detectou")
        face_appearance_[0] = face_appearance_[0] + 1
        if face_appearance_[0] >= 10:
            application.stop()
            face_appearance_[0] = 0
            print("CRIOU IMAGEM")
            cv2.imwrite('face_img.jpg', frame)
            # Read the binary file to send data to API
            body = ""
            filename = 'face_img.jpg'
            f = open(filename, "rb")
            body = f.read()
            f.close()
            application.start()
    else:
        face_appearance_[0] = 0


# load serialized model from disk
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt",
    "res10_300x300_ssd_iter_140000.caffemodel")

# initialize the video stream
# change src=0 to the camera source input
vs = VideoStream(src=0).start()
time.sleep(1.0)

# Calls detect_face every time_sec_ seconds
application = RepeatedTimer(time_sec_, detect_face, vs)
