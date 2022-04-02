from itertools import chain
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
from sklearn.model_selection import train_test_split

#print in green
def print_green(text):
    print("\033[92m", text, "\033[0m")

numbers = range(10)

uppercase_latin = range(10, 36)
uppercase_polish = range(36, 45)
uppercase = chain(uppercase_latin, uppercase_polish)

lowercase_latin = range(45, 71)
lowercase_polish = range(71, 80)
lowercase = chain(lowercase_latin, lowercase_polish)

letters = chain(uppercase, lowercase)
latin_letters = chain(uppercase_latin, lowercase_latin)
polish_letters = chain(uppercase_polish, lowercase_polish)

special_characters = range(80, 112)
whitespaces = range(112, 115)

non_letters = chain(numbers, special_characters, whitespaces)

full_dataset = range(115)

dataset_options = { "numbers": numbers,
    "uppercase_latin": uppercase_latin,
    "uppercase_polish": uppercase_polish,
    "uppercase": uppercase,
    "lowercase_latin": lowercase_latin,
    "lowercase_polish": lowercase_polish,
    "lowercase": lowercase,
    "letters": letters,
    "latin_letters": latin_letters,
    "polish_letters": polish_letters,
    "special_characters": special_characters,
    "whitespaces": whitespaces,
    "non_letters": non_letters,
    "full_dataset": full_dataset } 

#get --dataset or -d argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="dataset to use",
    choices=dataset_options.keys(), default="full_dataset")
parser.add_argument("--input", "-i", help="dataset folder",
    default="data/images")
args = parser.parse_args()

dataset = list(dataset_options[args.dataset])
dataset_folder = args.input

def classno_to_char(ind):
    polish_letters = list("ĄĆĘŁŃÓŚŹŻ")
    polish_miniscule_letters = list("ąćęłńóśźż")
    
    if ind < 10:
        return str(ind)
    elif ind < 36:
        return chr(ord('A') + ind - 10)
    elif ind < 45:
        return polish_letters[ind - 36]
    elif ind < 71:
        return chr(ord('a') + ind - 45)
    elif ind < 80:
        return polish_miniscule_letters[ind - 71]
    elif ind < 95:
        return chr(ord('!') + ind - 80)
    elif ind < 102:
        return chr(ord(':') + ind - 95)
    elif ind < 108:
        return chr(ord('[') + ind - 102)
    elif ind < 112:
        return chr(ord('{') + ind - 108)
    elif ind == 112:
        return ' '
    elif ind == 113:
        return '\t'
    else:
        return "\n"

def classno_to_charname(ind):
    char = classno_to_char(ind)
    if char == ' ':
        return 'space'
    elif char == '\t':
        return 'tab'
    elif char == '\n':
        return 'newline'
    else:
        return char

def read_images(folder=dataset_folder):
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array

    images_list=[]
    target_list=[]

    for subfolder in os.listdir(folder):
        for sub in dataset:
            #if directory does not exist, continue
            if not os.path.isdir(os.path.join(folder, subfolder, str(sub))):
                continue
            for im in next(os.walk(os.path.join(folder,subfolder,str(sub))))[2]:
                # load the image
                img = load_img(os.path.join(folder,subfolder,str(sub),im))
                # print("Original:", type(img))

                # convert to numpy array
                img_array = img_to_array(img)
                # print(img_array.shape)
                images_list.append(img_array)
                target_list.append(list(dataset).index(sub))

    images = np.array(images_list)
    target = np.array(target_list)
    return images, target

images, labels = read_images()

print_green(str(images.shape[0]) + " images loaded")

from sklearn.model_selection import train_test_split

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2)

train_ds = tf.data.Dataset.from_tensor_slices( (train_images, train_labels) )
test_ds =  tf.data.Dataset.from_tensor_slices( (test_images, test_labels) )
validation_ds =  tf.data.Dataset.from_tensor_slices( (validation_images, validation_labels) )

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()

print_green(f"Training data size: {train_ds_size}")
print_green(f"Test data size: {test_ds_size}")
print_green(f"Validation data size: {validation_ds_size}")

def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, [227,227] )
    return image, label

train_ds = (train_ds
            .map(process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=32, drop_remainder=True))
test_ds = (test_ds
            .map(process_images)
            .shuffle(buffer_size=test_ds_size)
            .batch(batch_size=32, drop_remainder=True))
validation_ds = (validation_ds
            .map(process_images)
            .shuffle(buffer_size=validation_ds_size)
            .batch(batch_size=32, drop_remainder=True))

def AlexNet():
    NUMBER_OF_CLASSES = 1000
    return keras.models.Sequential([
        keras.layers.Conv2D(name='conv1', filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(name='conv2', filters=256, kernel_size=(5,5), strides=1, activation='relu', padding="same", groups=2),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(name='conv3', filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(name='conv4', filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", groups=2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(name='conv5', filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", groups=2),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, name='fc6', activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, name='fc7', activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(NUMBER_OF_CLASSES, name='fc8', activation='softmax')
    ])

#creating the model
model = AlexNet()

net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1", allow_pickle=True).item()
