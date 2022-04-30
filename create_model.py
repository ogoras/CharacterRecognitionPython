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
from classno_conversions import classno_to_char, classno_to_charname

#print in green
def print_green(text):
    print("\033[92m", text, "\033[0m")

if __name__ == '__main__':
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
    parser.add_argument("--dataset", "-s", help="dataset to use",
        choices=dataset_options.keys(), default="full_dataset")
    parser.add_argument("--input", "-i", help="dataset folder",
        default="data/images")
    parser.add_argument("--epochs", "-e", help="number of epochs",
        default=50)
    parser.add_argument("--debug", "-d", help="debug mode",
        action="store_true")
    parser.add_argument("--analyze", "-a", help="analyze results",
        action="store_true")
    parser.add_argument("--output", "-o", help="output file",
        default="model" + time.strftime("%Y%m%d-%H%M%S") + ".h5")
    parser.add_argument("--subject", "-n", help="subject number for a specialized model")
    args = parser.parse_args()

    dataset = list(dataset_options[args.dataset])
    dataset_folder = args.input

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
                    if (args.debug and img_array.shape != (28,28,3)):
                        print(img_array.shape)
                    images_list.append(img_array)
                    target_list.append(list(dataset).index(sub))

        images = np.array(images_list)
        target = np.array(target_list)
        return images, target

    images, labels = read_images()

    print_green(str(images.shape[0]) + " images loaded")

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2)

    if(args.debug):
        print_green(f"Rozmiar danych uczących:      {train_images.shape}")
        print_green(f"Rozmiar danych walidujących:  {validation_images.shape}")
        print_green(f"Rozmiar danych testujących:   {test_images.shape}")

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

    model.get_layer('conv1').set_weights(net_data["conv1"])
    model.get_layer('conv2').set_weights(net_data["conv2"])
    model.get_layer('conv3').set_weights(net_data["conv3"])
    model.get_layer('conv4').set_weights(net_data["conv4"])
    model.get_layer('conv5').set_weights(net_data["conv5"])

    model.get_layer('fc6').set_weights(net_data["fc6"])
    model.get_layer('fc7').set_weights(net_data["fc7"])
    model.get_layer('fc8').set_weights(net_data["fc8"])

    new_model = tf.keras.models.Sequential(model.layers[:-5])
    new_model.summary()

    for l in new_model.layers:
        l.trainable = False

    numClasses = np.unique(train_labels).shape[0]
    print_green("Number of classes: " + str(numClasses))

    i = 1
    layers = new_model.layers + [
            keras.layers.Dense(500+35*i, activation='relu'),      
            keras.layers.Dropout(0.2+0.23*np.random.uniform(0,1)),  
            keras.layers.Dense(numClasses, activation='softmax')
        ]
        

    new_model_transfer = tf.keras.models.Sequential(layers)
    new_model_transfer.compile(loss='sparse_categorical_crossentropy', 
                            optimizer=tf.optimizers.SGD(lr=0.005), 
                            metrics=['accuracy'])
    new_model_transfer.summary()

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    #help(ImageDataGenerator)
    datagen = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.1
                                )
    datagen.fit(train_images)

    train_images_processed = []
    for img, lbl in zip(train_images, train_labels):
        i,l = process_images(img,lbl)
        train_images_processed.append(i)
    train_images_processed = np.array( train_images_processed )

    if(args.debug):
        print_green("Train images shape: " + str(train_images_processed.shape))

    history = new_model_transfer.fit(datagen.flow(train_images_processed, train_labels, batch_size=16), epochs=args.epochs, validation_data=validation_ds,
            validation_freq=1)

    #history = new_model_transfer.fit(train_ds, epochs=args.epochs, validation_data=validation_ds,
    #          validation_freq=1)

    #save the model to the file
    new_model_transfer.save(os.path.join("models/", args.output))

    if(args.analyze):
        f,ax = plt.subplots(1,2, dpi=150, figsize=(9,3))
        ax[0].plot(history.history['loss'], label='Test Loss')
        ax[0].plot(history.history['val_loss'], label='Validate Loss')
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(history.history['accuracy'], label='Test Accuracy')
        ax[1].plot(history.history['val_accuracy'], label='Validate Accuracy')
        ax[1].set_title("Accuracy")
        ax[1].legend()


        new_model_transfer.evaluate(test_ds)

        new_model.predict(test_ds)

        def debug(msg):
            if (args.debug):
                print(msg)

        images = []
        y_true = []
        y_pred = []
        for batch in test_ds:
            pred = new_model_transfer.predict(batch[0])
            print(batch[1])
            print(pred)
            for idx, (prawdopodobienstwo, oczekiwane) in enumerate(zip(pred, batch[1])):
                debug(f"oczekiwana: {oczekiwane} obliczona : {np.argmax(prawdopodobienstwo)}")

                images.append(batch[0][idx,:,:,:]) # zapamietaj skojarzony obrazek
                y_true.append(oczekiwane)          # zapamiętaj wartosć rzeczywistą
                y_pred.append(np.argmax(prawdopodobienstwo)) # zapamiętaj pozycję z największym prawdopodobieństwem

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        confusion_matrix(y_pred, y_true)

        print(classification_report(y_pred, y_true))

        blednie_sklasyfikowane = np.where( y_true != y_pred )

        idx = np.where( y_true == 0 )
        y_true_hot_one = np.zeros( (y_true.shape[0], numClasses) )
        for idx, c in enumerate(y_true):
            y_true_hot_one[idx, c] = 1.0

        y_pred_keras = new_model_transfer.predict((np.array(images)))

        print(f"Rozmiar macierzy zawierającego zawierającej prawdopodobieństwa przynależności do danej klasy: {y_pred_keras.shape}")
        print(f"{'-'*110}")

        res = []
        print("Klasy oczekiwane, oraz prawdopodobieństwa obliczone dla maksymalnej klasy w formie listy.")
        for y_t, y_p in zip(y_true, y_pred_keras):
            y_p_sorted = np.sort(y_p)
            res.append( (y_t, np.argmax(y_p), y_p_sorted[-1], y_p_sorted[-2]) )
            #print(f"Klasa rzeczywista: {y_t:4}, Klasa obliczona (nr maksymalnego P_max): {np.argmax(y_p)}, P_max: {y_p_sorted[-1]:.2f}, P_max_2: {y_p_sorted[-2]:.2f}")

        # używamy Pandas i Dataframe tylko po to aby ładnie wyświetlić wyniki    
        df = pd.DataFrame(data=res, columns=["Klasa rzeczywista", "Klasa obliczona (nr. max. P_max)", "P_max", "P_max_2"])
        pd.set_option('display.max_rows', 100)
        df = df.sort_values("Klasa rzeczywista")
        print(df)

        plt.figure(1, dpi=200)
        for i in range(numClasses):
            if (np.sum(y_true_hot_one[:,i]) > 0):
                fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true_hot_one[:,i], y_pred_keras[:,i])
                auc_keras = auc(fpr_keras, tpr_keras)
                print(f"AUC dla klasy: {i+1}: {auc_keras}")

                plt.plot(fpr_keras, tpr_keras, label=f'Klasa {i+1} (size: {np.sum(y_true_hot_one[:,i]).astype(np.uint)}) (auc = {auc_keras:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

        # ROC zbiorczy

        y_pred_keras = new_model_transfer.predict((np.array(images)))
        y_pred_keras.shape

        plt.rc('font', size=6) 
        plt.figure(1, dpi=200)
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true_hot_one.ravel(), y_pred_keras.ravel())
        auc_keras = auc(fpr_keras, tpr_keras)
        print(f"AUC zbiorczy: {auc_keras}")

        plt.plot(fpr_keras, tpr_keras, label=f'(size: {len(y_true)}) (auc = {auc_keras:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Krzywa ROC - zbiorcza')
        plt.legend(loc='best')
        plt.show()