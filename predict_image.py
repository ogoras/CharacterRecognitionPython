import tensorflow as tf
import os
from tensorflow import keras

def predict_image(img , model_filename): #img as numpy array
    #predict class of image using the model
    model = keras.models.load_model(model_filename)
    prediction = model.predict(img)
