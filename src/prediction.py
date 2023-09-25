from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D, Dense, Dropout
from keras.layers import Flatten
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

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

def predict(config):

    required_keys = [
        "test_forms.extracted_path",
        "pre_processing.cell_width",
        "pre_processing.cell_height",
        "pre_processing.num_classes",
        "model.model_path",
    ]
    check_config_keys(config, required_keys)

    form_path = config["test_forms"].get("extracted_path")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    num_classes = config["pre_processing"].get("num_classes")

    model_path = config["model"].get("model_path")


    model = build_model(num_classes, cell_width, cell_height)
    model.load_weights(model_path)

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ا', 'ب', 'پ', 'ت','ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ',
                'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل',
                'م', 'ن', 'و', 'ه', 'ی']

    intensity_bs = 0
    intensity_ms = 0
    intensity_phd = 0

    first_name = ''
    last_name = ''
    student_id = ''

    forms = glob.glob(form_path + "/*")
    for data_path in forms:
        test_path = glob.glob(data_path + '/*.jpg')
        for path in test_path:
            image_name = os.path.basename(path)
            image = load_img(path, target_size=(cell_width, cell_height))
            image = img_to_array(image) / 255.
            image = np.expand_dims(image, axis=0)
            if "ID" in image_name:
                if str(classes[np.argmax(model.predict(image)[0])]) != 'empty':
                    student_id += str(classes[np.argmax(model.predict(image)[0])]) + " "
            elif "FN" in image_name:
                if str(classes[np.argmax(model.predict(image)[0])]) != 'empty':
                    first_name += str(classes[np.argmax(model.predict(image)[0])])
            elif "LN" in image_name:
                if str(classes[np.argmax(model.predict(image)[0])]) != 'empty':
                    last_name += str(classes[np.argmax(model.predict(image)[0])])
            elif "BS" in image_name:
                for i in range(cell_width):
                    for j in range(cell_height):
                        intensity_bs += image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]
            elif "MS" in image_name:
                for i in range(cell_width):
                    for j in range(cell_height):
                        intensity_ms += image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]
            elif "PHD" in image_name:
                for i in range(cell_width):
                    for j in range(cell_height):
                        intensity_phd += image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]

        form_name = os.path.basename(data_path)
        print("Form: ", form_name)
        print("Student ID: ", student_id)
        print("First name: ", first_name)
        print("Last name: ", last_name)
        field = min(intensity_bs, intensity_ms, intensity_phd)
        if field == intensity_bs:
            print("field: ", "کارشناسی")
        elif field == intensity_ms:
            print("field: ", "کارشناسی ارشد")
        else:
            print("field: ", "دکتری")

def check_config_keys(cfg, required_keys):
    """
    Checks the config file and raise a value if there is a problem
    Args:
        cfg (omegaconf.dictconfig.DictConfig): A config file that is shared through 'hydra'
        required_keys (list): A list of required keys to be checked

    Raises:
        ValueError: if a key is missing
        ValueError: if a key is none
    """
    for key in required_keys:
        value = cfg
        for subkey in key.split("."):
            if subkey not in value:
                raise ValueError(f"Key '{key}' is missing in the configuration.")
            value = value[subkey]

        if value is None:
            raise ValueError(f"Value for key '{key}' is None in the configuration.")


def check_config_file(config_path):
    """
    checks if the config file exists and can be properly loaded
    Args:
        config_path (str): the path of the config file

    Raises:
        ValueError: if the config file is empty or invalid
        FileNotFoundError: if the config file was not found
        FileNotFoundError: if there was a problem in parsing data
        ValueError: if there was a problem in loading data

    Returns:
        dict: a dictionary containing the config file
    """
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        if config is None:
            raise ValueError("The YAML file is empty or invalid.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file 'config.yaml' was not found.")
    except yaml.YAMLError as e:
        raise FileNotFoundError(f"Error parsing the YAML configuration file:")
    except ValueError as e:
        raise ValueError(f"Error loading the configuration data:")
    else:
        # Configuration loaded successfully, you can access settings here
        print("Configuration loaded successfully.")
    return config

if __name__ == "__main__":
    config_path = "config/config.yaml"
    config = check_config_file(config_path)
    predict(config)
