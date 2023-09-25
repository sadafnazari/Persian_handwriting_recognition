from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential

def build_model(num_classes, data_width, data_height):
    """
    builds the model
    Args:
        num_classes (int): number of classes
        data_width (int): width of each image data
        data_height (int): height of each image data

    Returns:
        'keras.engine.sequential.Sequential': build model
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(data_width, data_height, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
