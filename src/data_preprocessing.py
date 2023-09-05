import glob
import os
import shutil

import cv2
import numpy as np
import yaml


def aruco_extraction(img):
    """
    Extracts the aruco signs from the given image, and selects the boundaries points of the form

    Args:
        img (numpy.ndarray): an image from the dataset

    Returns:
        numpy.ndarray or None: boundaries of the form
    """
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    # Detect the markers in the image
    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(
        img, dictionary, parameters=parameters
    )

    # Checks how many markers are detected
    if len(marker_corners) != 4:
        print("{} arucos detected instead of 4!".format(len(marker_corners)))
        return None

    # flatten the marker_corners array
    marker_corners = [mc[0] for mc in marker_corners]

    # sorts the corners based on the ids [30: top left, 31:bottom left, 32:bottom right, 33:top left]
    id_corner_combined = zip(marker_ids, marker_corners)
    sorted_combined = sorted(id_corner_combined, key=lambda x: x[0])
    sorted_corners = [x[1] for x in sorted_combined]

    # selects the boundaries clock wise(top left point of the top left marker,
    #                                   top right point of the top right marker,
    #                                   bottom right point of the bottom right marker,
    #                                   bottom left point of the bottom left marker)
    boundaries = np.array(
        [
            sorted_corners[0][0],
            sorted_corners[1][1],
            sorted_corners[3][2],
            sorted_corners[2][3],
        ],
        dtype=np.float32,
    )
    return boundaries


def form_extraction(img, corners, form_width, form_height):
    """
    Applies perspective to the image and extracts the form

    Args:
        img (numpy.ndarray): an image from the dataset
        corners (numpy.ndarray): position of the corners of the form
        form_width (int): width of the form
        form_height (int): height of the form

    Returns:
        numpy.ndarray: image of the extracted form
    """
    form_points = np.array(
        [(0, 0), (form_width, 0), (form_width, form_height), (0, form_height)]
    ).astype(np.float32)

    # applies perspective tranformation
    perspective_transformation = cv2.getPerspectiveTransform(corners, form_points)
    form = cv2.warpPerspective(
        img, perspective_transformation, (form_width, form_height)
    )
    return form


def cell_extraction(
    img,
    img_path,
    extracted_dataset_path,
    type,
    form_width,
    form_height,
    cell_width,
    cell_height,
):
    """
    Extracts cells and the saves them based on the type of the given form

    Args:
        img (numpy.ndarray): an image from the dataset
        img_path (str): path of the image
        extracted_dataset_path (str): path of the directory for the final data
        type (str): type of the form, either 'a' or 'b'
        form_width (int): width of the form
        form_height (int): height of the form
        cell_width (int): width of each cell
        cell_height (int): height of each cell
    """
    # Calculate cell dimensions based on the number of lines
    num_horizontal_lines = 21
    num_vertical_lines = 14
    cell_width = form_width // num_vertical_lines
    cell_height = form_height // num_horizontal_lines

    for row in range(num_horizontal_lines):
        if type == "a":  # the directory is named for the form 'a'
            if row < 2:  # cells for number '0' and '1'
                directory = str(row)
            elif row > 18:  # cells for number '2' and '3'
                directory = str(row - 17)
            else:
                directory = str(row + 8)  # cells for the first part of the letters

        elif type == "b":  # the directory is named for the form 'b'
            if row < 2:  # cells for number '3' and '4'
                directory = str(row + 4)
            elif row > 16:  # cells for number '6', '7', '8', and '9'
                directory = str(row - 11)
            else:  # cells for the second part of the letters
                directory = str(row + 25)

        for col in range(num_vertical_lines):
            # calculates the position of the cells
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            cell = img[y1:y2, x1:x2]
            cell = cv2.resize(cell, (cell_width, cell_height))

            # drops the cells that contain markers
            if row < 2 and col < 2:
                continue
            if row < 2 and col > 11:
                continue
            if row > 18 and col < 2:
                continue
            if row > 18 and col > 11:
                continue

            cv2.imwrite(
                extracted_dataset_path
                + "/"
                + directory
                + "/"
                + img_path[img_path.find("/" + type + "/") + 3 : -4]
                + "_"
                + str(col)
                + ".jpg",
                cell,
            )


def make_directories(path, num_classes):
    """
    Creates directory  and the needed subdirectories. 0-9 are correspondent to numbers and 10-41 are correspondent to the persian letters.

    Args:
        path (str): The name of the main directory for creating the directories
        num_classes (int): indicates the number of claases
    """
    # Creating the main directory
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")

    # Creating the subdirectories within the main directory
    for i in range(num_classes):
        if not os.path.exists(path + "/" + str(i)):
            os.makedirs(path + "/" + str(i))


def label_dataset(
    dataset_path,
    labeled_dataset_path,
    type,
    form_width,
    form_height,
    cell_width,
    cell_height,
):
    """labels dataset by extracting form and each cell and saving them to a folder that represents the class of each cell

    Args:
        dataset_path (str): path of the dataset
        labeled_dataset_path (str): path for storing the labeled dataset
        type (str): type of the form, either 'a' or 'b'
        form_width (int): width of the form
        form_height (int): height of the form
        cell_width (int): width of each cell
        cell_height (int): height of each cell
    """

    for image_path in glob.glob(dataset_path + "/" + type + "/*.*"):
        image = cv2.imread(image_path)
        corners = aruco_extraction(image)
        if corners is None:
            print(
                f"The image {image_path[image_path.find('/' + type + '/')+3:]} is dropped."
            )
            continue
        form = form_extraction(image, corners, form_width, form_height)
        cell_extraction(
            form,
            image_path,
            labeled_dataset_path,
            type,
            form_width,
            form_height,
            cell_width,
            cell_height,
        )


def finalise_dataset(
    labeled_dataset_path, final_dataset_path, num_classes, val_ratio, test_ratio
):
    """
    Splits dataset into train, val, and test set based on the given ratio
    Args:
        labeled_dataset_path (str): path of the labeled dataset
        final_dataset_path (str): path for finalised dataset
        num_classes (int): number of classes
        val_ratio (float): ratio for validation set
        test_ratio (float): ratio for test set
    """
    make_directories(final_dataset_path + "/train", num_classes)
    make_directories(final_dataset_path + "/val", num_classes)
    make_directories(final_dataset_path + "/test", num_classes)

    for cls in range(num_classes):
        all_file_names = os.listdir(labeled_dataset_path + "/" + str(cls))

        np.random.shuffle(all_file_names)
        train_file_names, val_file_names, test_file_names = np.split(
            np.array(all_file_names),
            [
                int(len(all_file_names) * (1 - val_ratio + test_ratio)),
                int(len(all_file_names) * (1 - test_ratio)),
            ],
        )

        train_file_names = [
            labeled_dataset_path + "/" + str(cls) + "/" + name
            for name in train_file_names.tolist()
        ]
        val_file_names = [
            labeled_dataset_path + "/" + str(cls) + "/" + name
            for name in val_file_names.tolist()
        ]
        test_file_names = [
            labeled_dataset_path + "/" + str(cls) + "/" + name
            for name in test_file_names.tolist()
        ]

        print("________________________")
        print("Class : ", cls)
        print("Total images: ", len(all_file_names))
        print("Training: ", len(train_file_names))
        print("Validation: ", len(val_file_names))
        print("Testing: ", len(test_file_names))

        # Copy-pasting images
        for name in train_file_names:
            shutil.copy(name, final_dataset_path + "/train/" + str(cls))

        for name in val_file_names:
            shutil.copy(name, final_dataset_path + "/val/" + str(cls))

        for name in test_file_names:
            shutil.copy(name, final_dataset_path + "/test/" + str(cls))


def check_config(cfg, required_keys):
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


def preproessing(config):
    """preprocess data

    Args:
        config (dict): the config file
    """
    required_keys = [
        "dataset.splitted",
        "dataset.labeled",
        "dataset.final",
        "variables.form_width",
        "variables.form_height",
        "variables.cell_width",
        "variables.cell_height",
        "variables.num_classes",
        "variables.val_ratio",
        "variables.test_ratio",
    ]

    check_config(config, required_keys)

    form_width = config["variables"].get("form_width")
    form_height = config["variables"].get("form_height")

    cell_width = config["variables"].get("cell_width")
    cell_height = config["variables"].get("cell_height")

    num_classes = config["variables"].get("num_classes")

    val_ratio = config["variables"].get("val_ratio")
    test_ratio = config["variables"].get("test_ratio")

    dataset_path = config["dataset"].get("splitted")

    labeled_dataset_path = config["dataset"].get("labeled")
    final_dataset_path = config["dataset"].get("final")

    make_directories(labeled_dataset_path, num_classes)

    # Extracting the forms of the type 'a'
    label_dataset(
        dataset_path,
        labeled_dataset_path,
        "a",
        form_width,
        form_height,
        cell_width,
        cell_height,
    )

    # Extracting the forms of the type 'b'
    label_dataset(
        dataset_path,
        labeled_dataset_path,
        "b",
        form_width,
        form_height,
        cell_width,
        cell_height,
    )

    print("dataset is extracted and labeled successfully.")
    return
    finalise_dataset(
        labeled_dataset_path, final_dataset_path, num_classes, val_ratio, test_ratio
    )

    print("dataset is split to train, val, and test successfully.")


if __name__ == "__main__":
    try:
        with open("config/data_preprocess.yaml", "r") as config_file:
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
        preproessing(config)
