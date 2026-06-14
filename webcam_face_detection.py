#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

# Load Model
caffeModel = r"D:\Opencv oct-Nov\module_14\model\res10_300x300_ssd_iter_140000.caffemodel"
config_file = r"D:\Opencv oct-Nov\module_14\model\deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(config_file, caffeModel)


# Face Detection Function
def detect_face(image, detection_threshold=0.7):

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104, 117, 123)
    )

    net.setInput(blob)

    detections = net.forward()

    faces = []

    img_h = image.shape[0]
    img_w = image.shape[1]

    for detection in detections[0, 0]:

        confidence = detection[2]

        if confidence >= detection_threshold:

            left   = int(detection[3] * img_w)
            top    = int(detection[4] * img_h)
            right  = int(detection[5] * img_w)
            bottom = int(detection[6] * img_h)

            face_w = right - left
            face_h = bottom - top

            faces.append((left, top, face_w, face_h))

    return np.array(faces)


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect Faces
    faces = detect_face(frame)

    # Draw Bounding Boxes
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Live Face Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

