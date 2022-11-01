import os
from gc import callbacks
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras import models
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.callbacks
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import datetime
from keras.callbacks import LearningRateScheduler


#Model configuration
batch_size = 100
no_epochs = 300
train_num = 21120
train_directory = 'rand0/3dgpr800MHz_rand0/train/'
val_num = 21120
val_directory = 'rand0/3dgpr800MHz_rand0/validation/'
learning_rate = 0.1
no_classes = 10
validation_split = 0.2
verbosity = 1

classes = {'ep1.0', 'ep3.0', 'ep4.0', 'ep5.0', 'ep9.0', 'pec'}
class_num = len(classes)

def step_decay(epoch):
    x = 0.1
    if epoch >= 100: x = 0.01
    if epoch >= 150: x = 0.001
    return x
lr_decay = LearningRateScheduler(step_decay)

train_datagen = image.ImageDataGenerator(
    height_shift_range = 0.01,
    width_shift_range = 0.2,
    horizontal_flip = True,
    rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size = (256,250),
    color_mode = 'rgb',
    classes = classes,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle = True
)

val_datagen = image.ImageDataGenerator(rescale = 1./255)
validation_generator = val_datagen.flow_from_directory(
    val_directory,
    target_size = (256,250),
    color_mode = 'rgb',
    classes = classes,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle = True
)

#Creat the model
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(256, 250,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(class_num, activation='softmax'))

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                    optimizer=tensorflow.keras.optimizers.SGD(lr=learning_rate),
                    metrics=['accuracy'])

logs = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1)

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_num//batch_size,
    callbacks = [tensorboard_callback, lr_decay],
    validation_data = validation_generator,
    validation_steps = batch_size,
    epochs = no_epochs
)

print(model.summary())
model.save_weights('weights/radar_cnn2d_weight'+ datetime.datetime.now().strftime("%Y%m%d-%H%M") + '.hdf5')