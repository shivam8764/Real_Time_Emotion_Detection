#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:14:19 2023

@author: nirbhaynair
"""
import os
os.environ['OPENCV_VIDEO_CAPTURE_USE_AVFOUNDATION'] = '1'
import cv2
import numpy as np
import keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import tensorflow as tf

from keras.models import model_from_json
from keras.preprocessing import image
os.chdir('/Users/nirbhaynair/Real Time Emotion Detection')

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer (1).h5')


## face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade-frontalface-default.xml')



cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5) # <- face is detected here 


    for (x,y,w,h) in faces_detected:

        offset = 50

        cv2.rectangle(test_img,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,0),thickness=7)# the rectangle on the face

        roi_gray=gray_img[y:y+h,x:x+w]#cropping region of interest i.e. face area from  image

        roi_gray=cv2.resize(roi_gray,(48,48))

        img_pixels = image.img_to_array(roi_gray)

        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x-offset), int(y+offset)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        #print(predicted_emotion)
    
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
