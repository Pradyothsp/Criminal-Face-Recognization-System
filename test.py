import glob
import os

import cv2
import numpy as np
from PIL import Image

quary_image_path = "anxiety-confused-indian-man-612308.jpg"

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer1/trainingData.yml")

img = cv2.imread(quary_image_path)

# cv2.imshow('image',img)

gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)

# print(gray)
faces = faceDetect.detectMultiScale(gray,1.3,5)

# print(faces)

file_path = []

for (x,y,w,h) in faces:
    # cv2.rectangle(img, (x,y), (x+w , y+h), (0,255,0), 2)
    id, conf = rec.predict(gray[y:y+h,x:x+w])
    print(id)

    file_path.extend(glob.glob("dataSet/User."+str(id)+".*"))

print(file_path)

for imagePath in file_path:
    image = cv2.imread(imagePath)
    
    # display the image on screen with imshow()
    # after checking that it loaded
    if image is not None:
        cv2.imshow(imagePath, image)
        cv2.waitKey(1000)
    elif image is None:
        print ("Error loading: " + imagePath)
        #end this loop iteration and move on to next image
        continue