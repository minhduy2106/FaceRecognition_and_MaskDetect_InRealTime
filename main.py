import cv2
import tensorflow as tf
from keras import models
import numpy as np

#load camera
cap = cv2.VideoCapture(1)

#load model
model = tf.keras.models.load_model('face_mask.h5')

#Resize img
img_size = 224
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 2


while True:
    ret, frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (255,255,255), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print('Face not detected.')
            #cv2.putText(frame, 'Face not detected', (60,200), font, 3, (0,0,255), 5)
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex : ex+ew]
    
    final_img = cv2.resize(face_roi, (img_size, img_size))
    final_img = np.expand_dims(final_img, axis=0)
    final_img = final_img/255.0
    Predict = model.predict(final_img)

    if(Predict < 0):
        status = 'No Mask! Please wear mask.'
        for (x,y,w,h) in faces:
            cv2.putText(frame, status, (x,y-20), font, font_scale, (0,0,255), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
    else :
        status = "Face Mask"
        for (x,y,w,h) in faces:
            cv2.putText(frame, status, (x,y-20), font, font_scale, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow('Detect Face Mask', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print('duy oc cho')