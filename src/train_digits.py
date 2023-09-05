from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D, Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
import yaml

def train_process(config):

def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(40, 40, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(40, 40, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def compile_model(model, optimizer, loss_type, metrics):
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model.summary()

def data_generator(rescale, shear_range, zoom_range, width_shift_range, height_shift_range, train_batch, val_batch)
# Data labeling and augmentation
datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        # rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
)
train_batch = 128
val_batch = 30
train_generator = datagen.flow_from_directory(
        'Train/',
        target_size=(40, 40),
        batch_size=train_batch,
        shuffle=True,
        class_mode='categorical')

val_generator = datagen.flow_from_directory(
        'Validation/',
        target_size=(40, 40),
        batch_size=val_batch,
        shuffle=False,
        class_mode='categorical')

train_generator.class_indices

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
checkpoint_loss = ModelCheckpoint(
    filepath="letters.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True)
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples//train_batch,
      validation_data=val_generator,
      validation_steps=val_generator.samples//val_batch,
      epochs=50,
      verbose=1,
      callbacks=checkpoint_loss)


if __name__ == "__main__":
    try:
        with open("config/model.yaml", "r") as config_file:
            config = yaml.safe_load(config_file)
        if config is None:
            raise ValueError("The YAML file is empty or invalid.")
    except FileNotFoundError:
        print("The configuration file 'config.yaml' was not found.")
    except yaml.YAMLError as e:
        print("Error parsing the YAML configuration file:")
        print(e)
    except ValueError as e:
        print("Error loading the configuration data:")
        print(e)
    else:
        # Configuration loaded successfully, you can access settings here
        print("Configuration loaded successfully.")
        train_process(config)
