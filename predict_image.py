import tensorflow as tf
import os
from tensorflow import keras
import numpy as np

def predict_image(img, model_filename): #img as numpy array
    #predict class of image using the model
    model = keras.models.load_model(model_filename)
    prediction = model.predict((img))
    #find the mode
    max_index = np.argmax(prediction)
    return max_index

def predict_image_from_file(img_filename, model_filename):
    #predict class of image using the model
    img = keras.preprocessing.image.load_img(img_filename, target_size=(227, 227, 3))
    img = keras.preprocessing.image.img_to_array(img)
    img = img.reshape(1, 227, 227, 3)
    return predict_image(img, model_filename)
