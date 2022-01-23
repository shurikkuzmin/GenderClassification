from json import load
import os
from sys import stdout
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow.keras
import argparse
from tensorflow.keras.models import load_model
import numpy as np


import cv2
plt.ion()

class AlexNet:
    @staticmethod
    def build():
        model = models.Sequential([
            # 1st Layer
            layers.Conv2D(96, (11,11), strides=(4,4), activation="relu", input_shape=(64,64,3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, strides=(2,2)),
            # 2nd Layer
            layers.Conv2D(256, (11,11), strides=(1,1), activation="relu", padding="same"),
            layers.BatchNormalization(),
            # 3rd Layer
            layers.Conv2D(384, (3,3), strides=(1,1), activation="relu", padding="same"),
            layers.BatchNormalization(),
            # 4th Layer
            layers.Conv2D(384, (3,3), strides=(1,1), activation="relu", padding="same"),
            layers.BatchNormalization(),
            # 5th Layer
            layers.Conv2D(256, (3,3), strides=(1,1), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, strides=(2,2)),
            # Flatten Layer
            layers.Flatten(),
            # Fully Connected Layer 1
            layers.Dense(4096, activation="relu"),
            layers.Dropout(0.5),
            # Fully Connected Layer 2
            layers.Dense(4096, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(  
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics="accuracy"
        )
        return model        

class MiniVGGNet:
    @staticmethod
    def build():
        model = models.Sequential([
             # 1st Big Layer
            layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(64,64,3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3,3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, strides=(2,2)),
            
            # 2nd Big Layer
            layers.Conv2D(64, (3,3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3,3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, strides=(2,2)),
            layers.Dropout(0.25),

            # 3rd Layer
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ])
        
        model.compile(  
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics="accuracy"
        )
        return model
class LossHistory(tensorflow.keras.callbacks.Callback):
    def __init__(self):
        super(LossHistory).__init__()
        self.accuracy = []
        self.val_accuracy = []
        self.loss = []
        self.val_loss = []
        self.epochs = []
    def on_epoch_end(self, epoch, logs):
        self.accuracy.append(logs["accuracy"])
        self.val_accuracy.append(logs["val_accuracy"])
        self.loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])
        self.epochs.append(epoch)

        if epoch >= 3:
            fig = plt.figure(1)
            fig.clf()
            plt.plot(self.epochs, self.accuracy, "g+-")
            plt.plot(self.epochs, self.loss, "ro-")
            plt.plot(self.epochs, self.val_accuracy,"b-")
            plt.plot(self.epochs, self.val_loss,"k--")
            plt.legend(["accuracy","loss","val_accuracy", "val_loss"])
            plt.ylim(0,2.0)
            plt.draw()
            plt.pause(0.01)

def train_model(weights_name: str) -> models.Sequential:
    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode="nearest")

    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory("Dataset/Train/",
                                                        batch_size=256, 
                                                        class_mode="binary",
                                                        target_size=(64, 64))
    validation_generator = validation_datagen.flow_from_directory("Dataset/Validation/",
                                                                  batch_size=256,
                                                                  class_mode="binary",
                                                                  target_size=(64, 64))
    #model = AlexNet.build()
    model = MiniVGGNet.build()
    tensorflow.keras.utils.plot_model(model,"minivggnet.png", show_shapes=True)

    callbacks = [LossHistory()]

    hist = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=256,
                    validation_steps=16,
                    callbacks=callbacks,
                    epochs=10)
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    model.save(weights_name)
    return model

def predict(model: models.Sequential):
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory("Dataset/Test/",
                                                       seed=2,
                                                       batch_size=10,
                                                       class_mode="binary",
                                                       target_size=(64, 64))
  
    original_images = test_generator[0][0]
    images = np.array([cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_RGB2BGR) for image in original_images])
    indices = test_generator.index_array[0:10]
    filepaths = np.array(test_generator.filepaths)[indices]
    
    prediction = np.array(np.ravel(model.predict(original_images)).astype(np.uint0))
    print("Prediction=", prediction)
    print("Reality=", test_generator[0][1])
    labels=["Woman", "Man"]


    for i in range(3):
        image = cv2.imread(filepaths[i])
        cv2.putText(image, labels[prediction[i]], org=(10,50), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, 
                    color=(255, 0, 0), thickness=3)
        cv2.imshow("Orig"+str(i+1),image)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train",
                       help="the name of trained model")
    group.add_argument("-l", "--load",
                       help="the name of the model to load")
    args = vars(parser.parse_args())
    
    if args["train"] is not None:
        model = train_model(args["train"])
    else:
        model = load_model(args["load"])
    
    predict(model)
    cv2.waitKey(0)

        
    
    
   