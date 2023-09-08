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
    required_keys = [
        "dataset.final",
        "pre_processing.cell_width",
        "pre_processing.cell_height",
        "pre_processing.num_classes",
        "data_generator.rescale",
        "data_generator.shear_range",
        "data_generator.zoom_range",
        "data_generator.width_shift_range",
        "data_generator.height_shift_range",
        "model.model_path",
        "model.train_batch",
        "model.val_batch",
        "model.test_batch",
        "model.epochs",
        "model.verbos"
    ]

    check_config_keys(config, required_keys)

    train_data, val_data, test_data = data_generator(config)
    model = build_model(num_classes)
    train_model(model, config, train_data, val_data)


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


def data_generator(train_path, val_path, test_path, train_batch, val_batch, test_batch, data_width, data_height, rescale, shear_range, zoom_range, width_shift_range, height_shift_range):
    # Data labeling and augmentation
    datagen = ImageDataGenerator(
                rescale=rescale,
                shear_range=shear_range,
                zoom_range=zoom_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
    )
    train_generator = datagen.flow_from_directory(
            train_path+"/",
            target_size=(data_width, data_height),
            batch_size=train_batch,
            shuffle=True,
            class_mode='categorical')

    val_generator = datagen.flow_from_directory(
            val_path+"/",
            target_size=(data_width, data_height),
            batch_size=test_batch,
            shuffle=False,
            class_mode='categorical')

    test_generator = datagen.flow_from_directory(
            test_path+"/",
            target_size=(data_width, data_height),
            batch_size=val_batch,
            shuffle=False,
            class_mode='categorical')

    print(train_generator.class_indices)
    print(val_generator.class_indices)
    print(test_generator.class_indices)
    return train_generator, val_generator, test_generator


def train_model(model, train_data, val_data, train_batch, val_batch, model_path, epochs, verbos):
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc'])
    checkpoint_loss = ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        verbose=verbos,
        save_best_only=True)
    history = model.fit(
        train_data,
        steps_per_epoch=train_data.samples//train_batch,
        validation_data=val_data,
        validation_steps=val_data.samples//val_batch,
        epochs=epochs,
        verbose=verbos,
        callbacks=checkpoint_loss)
    print("done")

def evaluate_model(model, test_data)


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
    train_process(config)
