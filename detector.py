import cv2
import numpy as np 

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer1/trainingData.yml")
id = 0
# font=cv2.cv2.putText(cv2.FONT_HERSHEY_SIMPLEX,5,1,0,4)

font = cv2.FONT_HERSHEY_SIMPLEX 
color = (255, 0, 0)

while True:    
    ret, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w , y+h), (0,255,0), 2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id == 1):
                id="Abhi"
        elif(id == 2):
                id="Pradhukz"
        elif(id == 3):
                id="sahil"
        cv2.putText(img,str(id),(x,y+h),font,1,color,2,cv2.LINE_AA)
    cv2.imshow('Face' , img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()