# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:40:04 2018

@author: Nate
"""
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import model_from_json

face_cascade = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('error')

model = model_from_json(open("C:/Users/Nate/Documents/python scripts/video_emotion_analysis/model/facial_expression_model_structure.json", "r").read())
model.load_weights('C:/Users/Nate/Documents/python scripts/video_emotion_analysis/model/facial_expression_model_weights.h5') #load weights

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emotion_counts = {}


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
		
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		  #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
		
        emotion = emotions[max_index]
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 1
        else:
            emotion_counts[emotion] += 1
        
        cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
    
    cv2.imshow('frame',frame)
    #cv2.imshow('grayF',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()

s = sum(emotion_counts.values())
for k, v in emotion_counts.items():
    pct = v * 100.0 / s
    print(k, pct)
