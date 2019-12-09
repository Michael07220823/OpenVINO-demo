import cv2

import imutils

import time

model_path = "emotions-recognition-retail-0003.xml"

pbtxt_path = "emotions-recognition-retail-0003.bin"

net = cv2.dnn.readNet(model_path, pbtxt_path)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cascade_scale = 1.2

cascade_neighbors = 6

minFaceSize = (30,30)

# Specify target device

net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

def getFaces(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor= cascade_scale,

        minNeighbors=cascade_neighbors,

        minSize=minFaceSize,

        flags=cv2.CASCADE_SCALE_IMAGE

    )

    bboxes = []

    for (x,y,w,h) in faces:

        if(w>minFaceSize[0] and h>minFaceSize[1]):

            bboxes.append((x, y, w, h))

    return bboxes

camera = cv2.VideoCapture(0)

frameID = 0

grabbed = True

start_time = time.time()

while grabbed:

    (grabbed, img) = camera.read()

    img = cv2.resize(img, (550,400))

    # Read an image

    out = []

    frame = img.copy()

    faces = getFaces(frame)

    x, y, w, h = 0, 0, 0, 0

    i = 0

    for (x,y,w,h) in faces:

        cv2.rectangle( frame,(x,y),(x+w,y+h),(255,255,255),1)

        if(w>0 and h>0):

            facearea = frame[y:y+h, x:x+w]

            # Prepare input blob and perform an inference

            blob = cv2.dnn.blobFromImage(facearea, size=(64, 64), ddepth=cv2.CV_8U)

            net.setInput(blob)

            out = net.forward()

            neutral = int(out[0][0][0][0] * 100)

            happy = int(out[0][1][0][0] * 100)

            sad = int(out[0][2][0][0] * 100)

            surprise = int(out[0][3][0][0] * 100)

            anger = int(out[0][4][0][0] * 100)

            yy = y

            line2 = "{}%\n{}%\n{}%\n{}%\n{}%".format(neutral,happy,sad,surprise,anger)

            y0, dy = yy, 35

            for ii, txt in enumerate(line2.split('\n')):

                y = y0 + ii*dy

                cv2.putText(frame, txt, (x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            line1 = "Neutral:\nHappy:\nSad:\nSurprise:\nAnger:"

            y0, dy = yy, 35

            for ii, txt in enumerate(line1.split('\n')):

                y = y0 + ii*dy

                cv2.putText(frame, txt, (x+55, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            i += 1

    # Save the frame to an image file

    cv2.imshow("FRAME", frame)

    frameID += 1

    fps = frameID / (time.time() - start_time)

    print("FPS:", fps)

    cv2.waitKey(1)
