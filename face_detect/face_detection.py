import cv2
import imutils
import time

model_path = "face-detection-adas-0001.xml"
pbtxt_path = "face-detection-adas-0001.bin"

net = cv2.dnn.readNet(model_path, pbtxt_path)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

camera = cv2.VideoCapture(0)
frameID = 0
grabbed = True
start_time = time.time()

while grabbed:
    (grabbed, img) = camera.read()
    img = cv2.resize(img, (900,640))
    frame = img.copy()

    # Prepare input blob and perform an inference
    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()
    
    # Draw detected faces on the frame
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])
        if confidence > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
    cv2.imshow("FRAME", frame)
    frameID += 1
    fps = frameID / (time.time() - start_time)
    print("FPS:", fps)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()