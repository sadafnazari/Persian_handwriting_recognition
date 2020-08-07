from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D, Dense
from keras.layers import Flatten
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import glob
from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import shutil

root_dir = ''
classes_dir = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
               '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
               '34', '35', '36', '37', '38', '39', '40', '41', '42']

val_ratio = 0.20
test_ratio = 0.05

for cls in classes_dir:
    try:
        os.makedirs('/Train/' + cls)
        os.makedirs('/Validation/' + cls)
        os.makedirs('/Test/' + cls)
    except:
        print("Directories already exist")


    # Creating partitions of the data after shuffeling
    src = root_dir + cls  # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * (1 - val_ratio + test_ratio)),
                                                               int(len(allFileNames) * (1 - test_ratio))])


    train_FileNames = [src+'/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('________________________')
    print('Class : ', cls)
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, '/Train/' + cls)

    for name in val_FileNames:
        shutil.copy(name, '/Validation/' + cls)

    for name in test_FileNames:
        shutil.copy(name, '/Test/' + cls)


# create model
num_classes = 33
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(40,40,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1, 1), activation='relu', input_shape=(40,40,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
model.summary()

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






