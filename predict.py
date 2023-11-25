import os
from keras.models import Model
from keras.models import load_model
from tkinter import filedialog
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import cv2
from tensorflow.image import resize
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

img = filedialog.askopenfilename()
model = load_model('vgg16_nft2.h5')

image1 = cv2.imread(img)
image1 = cv2.resize(image1, (224,224))
image1 = tf.expand_dims(image1,0)

answer = model.predict(image1)

list_ = answer[0].tolist()
step = list_.index(max(list_))
print(step)
