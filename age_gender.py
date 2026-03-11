import cv2
import numpy as np


faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"


faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

print("Models loaded successfully")

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']

genderList = ['Male', 'Female']


video = cv2.VideoCapture(0)
video.set(2,1280)
video.set(3,720)

if not video.isOpened():
    print("Camera not detected")
    exit()

while True:

    ret, frame = video.read()

    if not ret:
        break
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),[104,117,123], True, False)

    faceNet.setInput(blob)
    detections = faceNet.forward()

    h, w = frame.shape[:2]

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.7:

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x1,y1,x2,y2) = box.astype(int)

            padding = 20

            x1 = max(0, x1-padding)
            y1 = max(0, y1-padding)
            x2 = min(w-1, x2+padding)
            y2 = min(h-1, y2+padding)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            face = cv2.resize(face,(227,227))

            faceBlob = cv2.dnn.blobFromImage(
                face,
                1.0,
                (227,227),
                (78.426,87.768,114.896),
                swapRB=False
            )

            genderNet.setInput(faceBlob)
            genderPred = genderNet.forward()

            gender = genderList[genderPred[0].argmax()]
            gender_conf = genderPred[0].max()

            ageNet.setInput(faceBlob)
            agePred = ageNet.forward()

            age = ageList[agePred[0].argmax()]
            age_conf = agePred[0].max()

            label = f"{gender},{age}"

            cv2.putText(frame,label,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,255),2)

    cv2.imshow("Age and Gender Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()