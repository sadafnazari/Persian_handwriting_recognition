import yaml
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def train_process(config):
    """
    main function for the training process
    Args:
        config (dict): config file
    """
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
        "model.verbos",
    ]

    check_config_keys(config, required_keys)

    dataset_path = config["dataset"].get("final")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    num_classes = config["pre_processing"].get("num_classes")

    rescale = config["data_generator"].get("rescale")
    shear_range = config["data_generator"].get("shear_range")
    zoom_range = config["data_generator"].get("zoom_range")
    width_shift_range = config["data_generator"].get("width_shift_range")
    height_shift_range = config["data_generator"].get("height_shift_range")

    model_path = config["model"].get("model_path")
    train_batch = config["model"].get("train_batch")
    val_batch = config["model"].get("val_batch")
    test_batch = config["model"].get("test_batch")
    epochs = config["model"].get("epochs")
    verbos = config["model"].get("verbos")

    train_data, val_data, test_data = data_generator(
        dataset_path + "/train/",
        dataset_path + "/val/",
        dataset_path + "/test/",
        train_batch,
        val_batch,
        test_batch,
        cell_width,
        cell_height,
        rescale,
        shear_range,
        zoom_range,
        width_shift_range,
        height_shift_range,
        num_classes,
    )
    model = build_model(num_classes, cell_width, cell_height)
    train_model(
        model, train_data, val_data, train_batch, val_batch, model_path, epochs, verbos
    )
    evaluate_model(model, test_data)


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


def data_generator(
    train_path,
    val_path,
    test_path,
    train_batch,
    val_batch,
    test_batch,
    data_width,
    data_height,
    rescale,
    shear_range,
    zoom_range,
    width_shift_range,
    height_shift_range,
    num_classes,
):
    """
    creates data generators for the train, val, and test dataset
    Args:
        train_path (str): path of the train data
        val_path (str): path of the val data
        test_path (str): path fo the test data
        train_batch (int): batch size for train data
        val_batch (int): batch size for val data
        test_batch (int): batch size for test data
        data_width (int): width of each image
        data_height (int): height of each image
        rescale (float): rescale parameter
        shear_range (float): shear range parameter
        zoom_range (float): zoom range parameter
        width_shift_range (float): width shift range parameter
        height_shift_range (float): height shift range parameter
        num_classes (int): number of classes

    Returns:
        'keras.preprocessing.image.DirectoryIterator': train data generator
        'keras.preprocessing.image.DirectoryIterator': val data generator
        'keras.preprocessing.image.DirectoryIterator': test data generator
    """
    # Data labeling and augmentation
    datagen = ImageDataGenerator(
        rescale=rescale,
        shear_range=shear_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
    )
    # Create an empty dictionary to store class indices, this is because the mapping of classes should be in the correct order
    class_indices = {}
    for i in range(num_classes):
        class_indices[str(i)] = i

    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(data_width, data_height),
        batch_size=train_batch,
        shuffle=True,
        class_mode="categorical",
        classes=class_indices,
    )

    val_generator = datagen.flow_from_directory(
        val_path,
        target_size=(data_width, data_height),
        batch_size=test_batch,
        shuffle=False,
        class_mode="categorical",
        classes=class_indices,
    )

    test_generator = datagen.flow_from_directory(
        test_path,
        target_size=(data_width, data_height),
        batch_size=val_batch,
        shuffle=False,
        class_mode="categorical",
        classes=class_indices,
    )

    return train_generator, val_generator, test_generator


def train_model(
    model, train_data, val_data, train_batch, val_batch, model_path, epochs, verbos
):
    """
    trains the model based.
    Args:
        model (keras.engine.sequential.Sequential): the model
        train_data (keras.preprocessing.image.DirectoryIterator): training data generator
        val_data (keras.preprocessing.image.DirectoryIterator): validation data generator
        train_batch (int): batch size for train data
        val_batch (int): batch size for val data
        model_path (str): path of the model
        epochs (int): number of epochs
        verbos (int): verbos
    """
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    print(model.summary())

    checkpoint_loss = ModelCheckpoint(
        filepath=model_path, monitor="val_loss", verbose=verbos, save_best_only=True
    )
    print("Training begins...")
    history = model.fit(
        train_data,
        steps_per_epoch=train_data.samples // train_batch,
        validation_data=val_data,
        validation_steps=val_data.samples // val_batch,
        epochs=epochs,
        verbose=verbos,
        callbacks=checkpoint_loss,
    )
    print("Training is done.")


def evaluate_model(model, test_data):
    """
    evaluates the model
    Args:
        model (keras.engine.sequential.Sequential): the model
        test_data (keras.preprocessing.image.DirectoryIterator): test data generator
    """
    # evaluates the model by calculating loss and accuracy
    loss, accuracy = model.evaluate(test_data)

    # Print the evaluation results
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")


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
