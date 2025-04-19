import cv2
import os
import numpy as np

def faceBox(faceNet,frame):
    print(frame)
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame,bboxs

# Define the directory where model files are stored
MODEL_DIR = os.path.join(os.getcwd(), "models")

# Load models using relative paths
faceProto = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
faceModel = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(MODEL_DIR, "age_deploy.prototxt")
ageModel = os.path.join(MODEL_DIR, "age_net.caffemodel")
genderProto = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
genderModel = os.path.join(MODEL_DIR, "gender_net.caffemodel")

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES=(78.4263377603,87.7689143744,114.895847746)

ageList=['(0-2)','(4-6)','(8-12)','(15-20)','(21-30)','(38-43)','(45-52)','(55-60)','(60-100)']
genderList=['Male','Female']


video = cv2.VideoCapture(0)  # Open webcam

padding=20

while True:
    ret, frame = video.read()
    frame,bboxs=faceBox(faceNet,frame)
    for bbox in bboxs:
        #face=frame[bbox[1]:bbox[3],bbox[0:bbox[2]]]
        face=frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding,frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face,1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]

        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0],bbox[1]-30),(bbox[2],bbox[1]),(0,255,0),-1)
        cv2.putText(frame,label,(bbox[0], bbox[1]-10),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
   

    cv2.imshow("Age-Gender", frame)

    k = cv2.waitKey(1)  # Wait for a key press
    if k == ord('q'):   # Press 'q' to exit
        break

# Release resources after the loop ends
video.release()
cv2.destroyAllWindows()

