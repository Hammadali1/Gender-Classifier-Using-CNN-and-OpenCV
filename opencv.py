# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:58:47 2018

@author: hammad123
"""


from keras.models import model_from_json
from keras.models import load_model


classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("main.h5")
print("Loaded model from disk")

import cv2
 
import numpy as np
import keras
 
# capture video from webcam
cap = cv2.VideoCapture(0)
# initialize face detection
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
 
while(True):
    # read a frame. frame is in BGR( Blue, Green, Red) format
    ret, frame = cap.read()
 
    # convert the image to grayscale for performing face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
 
    # create a rgb frame since our model expects its input image
    # to be in RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
 
    cropped_faces = []
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # increase the width, height of face region by some pixels.
        # we want a bit more pixels rather than just the face
        extra_pad = 40
        x = max(1, x - extra_pad)
        y = max(1, y - extra_pad)
        width = min(frame.shape[1], x + w + extra_pad*2)
        height = min(frame.shape[0], y + h + extra_pad*2)
        cv2.rectangle(rgb_frame, (x, y), (width, height), (0, 255, 255), 2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=rgb_frame[y:y+h,x:x+w]
        # crop the face
        cropped = rgb_frame[y: height, x: width].astype("float32")
        cropped = cv2.resize(cropped, (64, 64))
       
        # need to rescale the values from 0 to 1
        cropped = cropped /255.0
        cropped = np.clip(cropped, 0, 1)
        cropped_faces.append(cropped)
        eye=eyeCascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eye:
              cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
     
    # predict the faces
    cv2.putText(rgb_frame, "Total faces = {}".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (30, 255, 200), 2)
    if len(faces) > 0 and 1 == 1:
        predictions = classifier.predict(np.array(cropped_faces))
        #predictions = np.where(predictions.flatten() < 0.5, "female", "male")
        if(predictions.flatten()>0.5):
            predictions="male"
        else:
            predictions="female"
        for i, prediction in enumerate(predictions):
            print(prediction)
          # text = prediction
 
            x, y, _, _ = faces[i]
            cv2.putText(rgb_frame, prediction, (x, y),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
   
    # convert our RGB frame back to BGR format before displaying
    cv2.imshow('frame', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
