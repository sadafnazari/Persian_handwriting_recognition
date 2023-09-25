import glob
import math
import os

import cv2
import numpy as np
import yaml


def aruco_extraction(img):
    """
    Extracts the aruco signs from the given image, and selects the boundaries points of the form

    Args:
        img (numpy.ndarray): an image

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

    # corners based on the ids [34: top left, 35:top right, 36:bottom right, 33:bottom left]

    # selects the boundaries clock wise(top left point of the top left marker,
    #                                   top right point of the top right marker,
    #                                   bottom right point of the bottom right marker,
    #                                   bottom left point of the bottom left marker)
    boundaries = np.array(
        [
            marker_corners[2][3],
            marker_corners[1][2],
            marker_corners[0][1],
            marker_corners[3][0],
        ],
        dtype=np.float32,
    )
    return boundaries


def form_extraction(img, corners, form_width, form_height):
    """
    Applies perspective to the image and extracts the form

    Args:
        img (numpy.ndarray): an image
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
    img, img_path, extracted_path, form_width, form_height, cell_width, cell_height
):
    """
    Extracts cells and the saves them based on the type of the given form

    Args:
        img (numpy.ndarray): an image from the test forms
        img_path (str): path of the image
        extracted_path (str): path of the directory for the final data
        form_width (int): width of the form
        form_height (int): height of the form
        cell_width (int): width of each cell
        cell_height (int): height of each cell
    """
    image_name = os.path.basename(img_path)
    directory_name = os.path.splitext(image_name)[0]
    directory = extracted_path + "/" + directory_name
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Folder '{directory}' created.")
    else:
        print(f"Folder '{directory}' already exists.")
    # Locations are guessed TODO: Find a proper way to apply this
    starting_points = {"ID": (40, 305), "First Name": (40, 435), "Last Name": (40, 570)}
    ending_points = {
        "ID": (580, 385),
        "First Name": (580, 515),
        "Last Name": (580, 650),
    }
    degree_points = {"PHD": (75, 720), "MS": (225, 720), "BS": (450, 720)}

    number_of_cells = 8

    for i in range(number_of_cells):
        width = (ending_points["ID"][0] - starting_points["ID"][0]) // 8
        x1 = i * width + starting_points["ID"][0]
        y1 = starting_points["ID"][1]
        x2 = x1 + width
        y2 = ending_points["ID"][1]
        cell = img[y1:y2, x1:x2]
        cell = cv2.resize(cell, (cell_width, cell_height))
        cv2.imwrite(directory + "/" + "ID" + "_" + str(i) + ".jpg", cell)

    for i in range(number_of_cells):
        width = (ending_points["First Name"][0] - starting_points["First Name"][0]) // 8
        x1 = i * width + starting_points["First Name"][0]
        y1 = starting_points["First Name"][1]
        x2 = x1 + width
        y2 = ending_points["First Name"][1]
        cell = img[y1:y2, x1:x2]
        cell = cv2.resize(cell, (cell_width, cell_height))
        cv2.imwrite(directory + "/" + "FN" + "_" + str(i) + ".jpg", cell)

    for i in range(number_of_cells):
        width = (ending_points["Last Name"][0] - starting_points["Last Name"][0]) // 8
        x1 = i * width + starting_points["Last Name"][0]
        y1 = starting_points["Last Name"][1]
        x2 = x1 + width
        y2 = ending_points["Last Name"][1]
        cell = img[y1:y2, x1:x2]
        cell = cv2.resize(cell, (cell_width, cell_height))
        cv2.imwrite(directory + "/" + "LN" + "_" + str(i) + ".jpg", cell)

    x1_bs = degree_points["BS"][0]
    y1_bs = degree_points["BS"][1]
    x2_bs = degree_points["BS"][0] + 25
    y2_bs = degree_points["BS"][1] + 25
    cell = img[y1_bs:y2_bs, x1_bs:x2_bs]
    cell = cv2.resize(cell, (cell_width, cell_height))
    cv2.imwrite(directory + "/" + "BS" + ".jpg", cell)

    x1_ms = degree_points["MS"][0]
    y1_ms = degree_points["MS"][1]
    x2_ms = degree_points["MS"][0] + 25
    y2_ms = degree_points["MS"][1] + 25
    cell = img[y1_ms:y2_ms, x1_ms:x2_ms]
    cell = cv2.resize(cell, (cell_width, cell_height))
    cv2.imwrite(directory + "/" + "MS" + ".jpg", cell)

    x1_phd = degree_points["PHD"][0]
    y1_phd = degree_points["PHD"][1]
    x2_phd = degree_points["PHD"][0] + 25
    y2_phd = degree_points["PHD"][1] + 25
    cell = img[y1_phd:y2_phd, x1_phd:x2_phd]
    cell = cv2.resize(cell, (cell_width, cell_height))
    cv2.imwrite(directory + "/" + "PHD" + ".jpg", cell)


def extract_and_save(
    test_form_path, extracted_path, form_width, form_height, cell_width, cell_height
):
    """
    Extracts forms from the images and then extracts the cells and saves the results.

    Args:
        test_form_path (str): path of the test form
        extracted_path (str): path for saving the extracted cells
        form_width (int): width of the form
        form_height (int): height of the form
        cell_width (int): width of each cell
        cell_height (int): height of each cell
    """
    for image_path in glob.glob(test_form_path + "/*.*"):
        image = cv2.imread(image_path)
        corners = aruco_extraction(image)
        if corners is None:
            print(f"The image {image_path} is dropped.")
            continue
        form = form_extraction(image, corners, form_width, form_height)
        cell_extraction(
            form,
            image_path,
            extracted_path,
            form_width,
            form_height,
            cell_width,
            cell_height,
        )


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


def data_extraction(config):

    required_keys = [
        "test_forms.test_form_path",
        "test_forms.extracted_path",
        "pre_processing.form_width",
        "pre_processing.form_height",
        "pre_processing.cell_width",
        "pre_processing.cell_height",
    ]

    check_config_keys(config, required_keys)

    test_form_path = config["test_forms"].get("test_form_path")
    extracted_path = config["test_forms"].get("extracted_path")

    form_width = config["pre_processing"].get("form_width")
    form_height = config["pre_processing"].get("form_height")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    if not os.path.exists(extracted_path):
        os.makedirs(extracted_path)
        print(f"Folder '{extracted_path}' created.")
    else:
        print(f"Folder '{extracted_path}' already exists.")

    extract_and_save(
        test_form_path, extracted_path, form_width, form_height, cell_width, cell_height
    )


if __name__ == "__main__":
    config_path = "config/config.yaml"
    config = check_config_file(config_path)
    data_extraction(config)
