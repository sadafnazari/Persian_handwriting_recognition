from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D, Dense
from keras.layers import Flatten
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os


def main():
    # create model for letters
    model = Sequential()
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            input_shape=(60, 60, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            input_shape=(60, 60, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(40, activation="relu"))
    model.add(Dense(60, activation="relu"))
    model.add(Dense(60, activation="relu"))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(42, activation="softmax"))


    intensity_bs = 0
    intensity_ms = 0
    intensity_phd = 0
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ا', 'ب', 'پ', 'ت','ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ',
                'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل',
                'م', 'ن', 'و', 'ه', 'ی','empty']


    model.load_weights("models/trained_model.h5")
    data_path = "data/test_forms/extracted/test1/"
    test_path = glob.glob(data_path + '*.jpg')
    test_path.sort()
    first_name = ''
    last_name = ''
    student_id = ''
    for path in test_path:
        image_name = os.path.basename(path)
        image = load_img(path, target_size=(60, 60))
        image = img_to_array(image) / 255.
        image = np.expand_dims(image, axis=0)
        if image_name[0] == 'I':
            if str(number[np.argmax(model.predict(image)[0])]) != 'empty':
                student_id += str(np.argmax(model.predict(image)[0])) + " "
        elif image_name[0] == 'F':
            if str(number[np.argmax(model.predict(image)[0])]) != 'empty':
                first_name += str(number[np.argmax(model.predict(image)[0])])
        elif image_name[0] == 'L':
            if str(number[np.argmax(model.predict(image)[0])]) != 'empty':
                last_name += str(number[np.argmax(model.predict(image)[0])])
        elif image_name[0] == 'B':
            for i in range(40):
                for j in range(40):
                    intensity_bs += image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]
        elif image_name[0] == 'M':
            for i in range(40):
                for j in range(40):
                    intensity_ms += image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]
        elif image_name[0] == 'P':
            for i in range(40):
                for j in range(40):
                    intensity_phd += image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]

    print("******************")
    print(student_id)
    print(first_name)
    print(last_name)
    maximum = min(intensity_bs, intensity_ms, intensity_phd)
    if maximum == intensity_bs:
        print("کارشناسی")
    elif maximum == intensity_ms:
        print("کارشناسی ارشد")
    else:
        print("دکتری")

if __name__ == "__main__":
    main()
