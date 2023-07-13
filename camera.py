import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

facetracker = load_model('models/facetracker3.h5')
iristracker = load_model('models/iris.h5')
age_gender = load_model('models/age_gender.h5')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_face = tf.image.resize(rgb, (120, 120))
    resized_iris = tf.image.resize(rgb, (250, 250))
    resized_ag = tf.image.resize(rgb, (200, 200))

    # Face Tracker
    yhat_face = facetracker.predict(np.expand_dims(resized_face / 255, 0))
    confidence_face = yhat_face[0][0]
    sample_coords_face = yhat_face[1][0]

    # Iris Tracker
    yhat_iris = iristracker.predict(np.expand_dims(resized_iris / 255, 0))
    sample_coords_iris = yhat_iris[0, :4]

    cv2.circle(frame, tuple(np.multiply(sample_coords_iris[:2], [450,450]).astype(int)), 3, (255,0,0), -1)
    cv2.circle(frame, tuple(np.multiply(sample_coords_iris[2:], [450,450]).astype(int)), 3, (0,255,0), -1)  

    # Age Gender Prediction
    yhat_ag = age_gender.predict(np.expand_dims(resized_ag / 255, 0))
    age = yhat_ag[0]
    gender = yhat_ag[1]

    age_text = "age: {:.1f}".format(age[0][0])
    gender_text = "gender: {:.1f}".format(gender[0][0])
    cv2.putText(frame, age_text, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA) 
    cv2.putText(frame, gender_text, (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  
    
    if confidence_face > 0.9:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords_face[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords_face[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords_face[:2], [450, 450]).astype(int),
                                    [0, -30])),
                      tuple(np.add(np.multiply(sample_coords_face[:2], [450, 450]).astype(int),
                                    [80, 0])),
                      (255, 0, 0), -1)

        # Controls the text rendered
        label_text = "face ({:.4f})".format(float(confidence_face))
        cv2.putText(frame, label_text, tuple(np.add(np.multiply(sample_coords_face[:2], [450, 450]).astype(int),
                                                    [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('FaceTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
