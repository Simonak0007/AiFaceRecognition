import cv2
import time
from tensorflow import keras
import tensorflow as tf
import pathlib
import PIL
import PIL.Image
import os
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model('./dataset_fotky')
capture = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

class_names = ['Martin', 'adyas', 'dan', 'jakub', 'kuba', 'simon'   ]

while (True):
    ret, frame = capture.read()
    time.sleep(1)
    if ret:
        frame1 = frame
        frame = cv2.resize(frame, (180,180))
        img_array = tf.keras.preprocessing.image.img_to_array(frame)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions =  model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (30,100)
        fontScale              = 1.3
        fontColor              = (0,0,0)
        lineType               = 2
        print(predictions[0].shape)
        cv2.putText(frame1,"{} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(frame1)
        plt.subplot(1,2,2)
        plt.bar(class_names,predictions[0])
        plt.text(31,30,"{} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
        plt.show()
        cv2.imshow('video', frame1)

 
    if cv2.waitKey(1) == 27:
        break
 
capture.release()
 
cv2.destroyAllWindows()