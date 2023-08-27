from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D, Dense
from keras.layers import Flatten
import numpy as np
import glob
from keras.preprocessing.image import load_img, img_to_array

# create model for letters
num_classes_letters = 33
model_letters = Sequential()
model_letters.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(40,40,3)))
model_letters.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_letters.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1,1), activation='relu', input_shape=(40,40,3)))
model_letters.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_letters.add(Flatten())
model_letters.add(Dense(40, activation='relu'))
model_letters.add(Dense(60, activation='relu'))
model_letters.add(Dense(60, activation='relu'))
model_letters.add(Dense(40, activation='relu'))
model_letters.add(Dense(num_classes_letters, activation='softmax'))
model_letters.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
model_letters.summary()

# create model for digits
num_classes_digits = 11
model_digits = Sequential()
model_digits.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(40, 40, 3)))
model_digits.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_digits.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(40, 40, 3)))
model_digits.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_digits.add(Flatten())
model_digits.add(Dense(40, activation='relu'))
model_digits.add(Dense(60, activation='relu'))
model_digits.add(Dense(60, activation='relu'))
model_digits.add(Dense(40, activation='relu'))
model_digits.add(Dense(num_classes_digits, activation='softmax'))
model_digits.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
model_digits.summary()


# predicting!
intensity_bs = 0
intensity_ms = 0
intensity_phd = 0
alphabet = ['ا', 'ب', 'پ', 'ت','ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ',
            'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل',
            'م', 'ن', 'و', 'ه', 'ی', 'empty']
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'empty']


model_letters.load_weights("letters.h5")
model_digits.load_weights("digits.h5")
data_path = "test"
test_path = glob.glob(data_path + '*.jpg')
test_path.sort()
first_name = ''
last_name = ''
student_id = ''
for path in test_path:
    image = load_img(path, target_size=(40, 40))
    image = img_to_array(image) / 255.
    image = np.expand_dims(image, axis=0)
    if path[0] == 'I':
        if(str(number[np.argmax(model_digits.predict(image)[0])])) != 'empty':
            student_id += str(np.argmax(model_digits.predict(image)[0]))
    elif path[0] == 'F':
        if str(alphabet[np.argmax(model_letters.predict(image)[0])]) != 'empty':
            first_name += str(alphabet[np.argmax(model_letters.predict(image)[0])])
        elif path[0] == 'L':
            if str(str(alphabet[np.argmax(model_letters.predict(image)[0])])) != 'empty':
                last_name += str(alphabet[np.argmax(model_letters.predict(image)[0])])
    elif path[0] == 'B':
        for i in range(40):
            for j in range(40):
                intensity_bs += image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]
    elif path[0] == 'M':
        for i in range(40):
            for j in range(40):
                intensity_ms += image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]
    elif path[0] == 'P':
        for i in range(40):
            for j in range(40):
                intensity_phd += image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]


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