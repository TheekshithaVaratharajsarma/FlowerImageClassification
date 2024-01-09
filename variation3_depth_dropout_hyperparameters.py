#!/usr/bin/env python
# coding: utf-8

from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import shutil
from keras.layers.core import Dropout

class flower_recognition_model_var3:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=(3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(150))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        model.summary()
        return model

# Hyperparameters
no_epochs = 40
batch_size = 20
valid_split = 0.2
verbose = 1
optimizer = Adam()

# Image and class information
img_rows, img_cols = 224, 224
no_classes = 5  
input_shape = (img_rows, img_cols, 3)  # RGB

# Setting the path to data
data_dir = 'data'
base_dir = 'flower_split'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

#creating base directory & train, test directories
os.makedirs(base_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Splitting data into train and test sets
for flower_class in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, flower_class)
    images = os.listdir(class_dir)
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # Moving images to train directory
    for img in train_images:
        src = os.path.join(class_dir, img)
        dest = os.path.join(train_dir, flower_class, img)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)

    # Moving images to test directory
    for img in test_images:
        src = os.path.join(class_dir, img)
        dest = os.path.join(test_dir, flower_class, img)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)

# Data augmentation to increase diversity
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Building and compiling the model
model = flower_recognition_model_var3.build(input_shape=input_shape, classes=no_classes)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Using flow_from_directory to load and preprocess images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=no_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    verbose=verbose
)

train_loss = history.history['loss']
train_accuracy = history.history['accuracy']

# Evaluating the model on the test set
score = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size, verbose=verbose)
test_loss = score[0]
test_accuracy = score[1]

with open('training_history_var3.txt', 'w') as f:
    formatted_history = {key: [round(value, 4) for value in values] for key, values in history.history.items()}
    f.write(str(formatted_history))

with open('evaluation_output_var3.txt', 'w') as f:
    f.write(f'Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}') 