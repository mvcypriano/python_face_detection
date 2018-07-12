# python3 DetectFaceThread.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
from imutils.video import VideoStream
from RepeatedTimer import RepeatedTimer
import numpy as np
import argparse
import imutils
import time
import cv2
import threading
import time

# construct the argument parse and parse the arguments
confidence_ = 0.8
face_size_ = 200
time_sec_ = 0.2
face_appearance_ = [0];



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

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        if (endX - startX) < face_size_ or (endY - startY) < face_size_:
            continue

        relevant_face_counter = relevant_face_counter + 1
        print("Detectou")

    if relevant_face_counter == 1:
        face_appearance_[0] = face_appearance_[0] + 1
        if face_appearance_[0] >= 10:
            face_appearance_[0] = 0
            print("DETECTOU")
    else:
        face_appearance_[0] = 0


    if application.is_canceled:
        vs.stop()
        cv2.destroyAllWindows()
        application.stop()

# do a bit of cleanup
#cv2.destroyAllWindows()


# load serialized model from disk
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt",
    "res10_300x300_ssd_iter_140000.caffemodel")

# initialize the video stream
# change src=0 to the camera source input
vs = VideoStream(src=0).start()
time.sleep(1.0)

application = RepeatedTimer(time_sec_, detect_face, vs)
