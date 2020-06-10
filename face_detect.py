##########################face and eyes detection python code ##################################


import cv2
import numpy as np
#eye_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.8/dist-packages/cv2/data/haarcascade_eye.xml')
face_detect=cv2.CascadeClassifier('/usr/local/lib/python3.8/dist-packages/cv2/data/haarcascade_frontalface_default.xml')
img=cv2.imread('personal.jpg')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converting image into gray format 

faces=face_detect.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)#with detect multiscale perform actual operation
#eyes = eye_cascade.detectMultiScale(gray, 1.03, 5)

#drawing rectangle on detected face
for(x,y,w,h) in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3) #change coloum with (0,255,0),2
cv2.imwrite('personal.jpg',img)#python will rewrite with new image 
