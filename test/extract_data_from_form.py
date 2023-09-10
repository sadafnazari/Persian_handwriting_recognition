import numpy as np
import cv2
import math
import os
import glob
import yaml


def aruco_extraction(img):
    """
    Extracts the aruco signs from the given image, and selects the boundaries points of the form

    Args:
        img (numpy.ndarray): an image from the test forms

    Returns:
        numpy.ndarray or None: boundaries of the form
    """
    # Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()
    # Detect the markers in the image
    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
    # print("{} ArUco marker were detected.".format(len(marker_corners)))
    assert len(marker_corners) == 4, "You have detected {} instead of 4 ArUco markers!".format(len(marker_corners))
    p1 = (0, 0)
    p2 = (0, 0)
    p3 = (0, 0)
    p4 = (0, 0)
    # set the corners based on their ids
    for m in range(len(marker_ids)):
        if marker_ids[m] == 34:
            p1 = marker_corners[m][0][3][0], marker_corners[m][0][3][1]
        elif marker_ids[m] == 36:
            p2 = marker_corners[m][0][1][0], marker_corners[m][0][1][1]
        elif marker_ids[m] == 35:
            p3 = marker_corners[m][0][2][0], marker_corners[m][0][2][1]
        else:
            p4 = marker_corners[m][0][0][0], marker_corners[m][0][0][1]
    # print(p1, p2, p3, p4)
    return p1, p2, p3, p4

def form_extraction(img, corners, form_width, form_height):
    """
    Applies perspective to the image and extracts the form

    Args:
        img (numpy.ndarray): an image from the test forms
        corners (numpy.ndarray): position of the corners of the form
        form_width (int): width of the form
        form_height (int): height of the form

    Returns:
        numpy.ndarray: image of the extracted form
    """
    # extract the form to a 800 * 800 picture and correct its perspective
    print(corners)
    points1 = corners
    n = 800
    m = 800
    J = np.zeros((m, n))
    cv2.imshow('Original image', img)
    cv2.waitKey(0)
    points2 = np.array([(0, 0),
                        (n, 0),
                        (n, m),
                        (0, m)]).astype(np.float32)
    print(points1)
    print(points2)
    print(points1.dtype, points2.dtype)
    #H = np.array([[ 1.03459088e+00,  5.70022525e-03, -3.72709228e+01],
    #              [-3.10793092e-02,  1.08925579e+00, -2.01513801e+02],
    #             [-5.94946334e-05, -1.04173685e-05,  1.00000000e+00]])
    H = cv2.getPerspectiveTransform(points1, points2)
    output_size = (J.shape[1], J.shape[0])
    print(output_size, H)
    J = cv2.warpPerspective(img, H, output_size)
    cv2.imshow('Form', J)
    print("1234")
    cv2.waitKey(0)
    return J

def cell_extraction(
    img,
    img_path,
    extracted_path,
    form_width,
    form_height,
    cell_width,
    cell_height,
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
    # Calculate cell dimensions based on the number of lines
        # extract each letter, number and checkbox
    return
    I = img
    G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # first extracting edges
    E = cv2.Canny(G, 50, 120)
    # then find the corners by Harris
    G = np.float32(E)
    window_size = 4
    soble_kernel_size = 5
    alpha = 0.04
    H = cv2.cornerHarris(G, window_size, soble_kernel_size, alpha)
    H = H / H.max()
    M = np.uint8(H > 0.005) * 255

    # guess the location of the principal points
    starting_points = {'ID': (40, 200), 'First Name': (40, 310), 'Last Name': (40, 400)}
    ending_points = {'ID': (585, 280), 'First Name': (585, 372), 'Last Name': (585, 470)}
    degree_points = {'PHD': (75, 510), 'MS': (220, 510), 'BS': (450, 510)}

    # # extracting!
    # # Student ID
    # starting_flag = True
    # ending_flag = True
    # for i in range(800):
    #     if starting_flag or ending_flag:
    #         for j in range(800):
    #             if starting_flag or ending_flag:
    #                 if M[i][j] == 255:
    #                     if calculate_distance(starting_points['ID'], (j, i)) <= 13.0 and starting_flag:
    #                         starting_points['ID'] = (j, i)
    #                         # cv2.circle(img, (starting_points['ID']), 5, (0, 0, 255), -1)
    #                         starting_flag = False
    #                     if calculate_distance(ending_points['ID'], (j, i)) <= 13.0 and ending_flag:
    #                         ending_points['ID'] = (j, i)
    #                         # cv2.circle(img, (ending_points['ID']), 5, (0, 0, 255), -1)
    #                         ending_flag = False

    # # Student First Name
    # starting_flag = True
    # ending_flag = True
    # for i in range(800):
    #     if starting_flag or ending_flag:
    #         for j in range(800):
    #             if starting_flag or ending_flag:
    #                 if M[i][j] == 255:
    #                     if calculate_distance(starting_points['First Name'], (j, i)) <= 13.0 and starting_flag:
    #                         starting_points['First Name'] = (j, i)
    #                         # cv2.circle(img, (starting_points['First Name']), 5, (0, 0, 255), -1)
    #                         starting_flag = False
    #                     if calculate_distance(ending_points['First Name'], (j, i)) <= 13.0 and ending_flag:
    #                         ending_points['First Name'] = (j, i)
    #                         # cv2.circle(img, (ending_points['First Name']), 5, (0, 0, 255), -1)
    #                         ending_flag = False

    # # Student Last Name
    # starting_flag = True
    # ending_flag = True
    # for i in range(800):
    #     if starting_flag or ending_flag:
    #         for j in range(800):
    #             if starting_flag or ending_flag:
    #                 if M[i][j] == 255:
    #                     if calculate_distance(starting_points['Last Name'], (j, i)) <= 13.0 and starting_flag:
    #                         starting_points['Last Name'] = (j, i)
    #                         # cv2.circle(img, (starting_points['Last Name']), 5, (0, 0, 255), -1)
    #                         starting_flag = False
    #                     if calculate_distance(ending_points['Last Name'], (j, i)) <= 13.0 and ending_flag :
    #                         ending_points['Last Name'] = (j, i)
    #                         # cv2.circle(img, (ending_points['Last Name']), 5, (0, 0, 255), -1)
    #                         ending_flag = False

    # # Student Degree BS
    # point = degree_points['BS']
    # flag = True
    # for i in range(800):
    #     if flag:
    #         for j in range(800):
    #             if flag:
    #                 if M[i][j] == 255:
    #                     if calculate_distance(point, (j, i)) <= 10.0:
    #                         # cv2.circle(img, (j, i), 5, (255, 0, 0), -1)
    #                         degree_points['BS'] = (j, i)
    #                         flag = False
    # # Student Degree MS
    # point = degree_points['MS']
    # flag = True
    # for i in range(800):
    #     if flag:
    #         for j in range(800):
    #             if flag:
    #                 if M[i][j] == 255:
    #                     if calculate_distance(point, (j, i)) <= 15.0:
    #                         # cv2.circle(img, (j, i), 5, (255, 0, 0), -1)
    #                         degree_points['MS'] = (j, i)
    #                         flag = False
    # # Student Degree PHD
    # point = degree_points['PHD']
    # flag = True
    # for i in range(800):
    #     if flag:
    #         for j in range(800):
    #             if flag:
    #                 if M[i][j] == 255:
    #                     if calculate_distance(point, (j, i)) <= 15.0:
    #                         # cv2.circle(img, (j, i), 5, (255, 0, 0), -1)
    #                         degree_points['PHD'] = (j, i)
    #                         flag = False


    #     for i in range(8):
    #     w = ending_points['ID'][0] - starting_points['ID'][0]-10
    #     h = ending_points['ID'][1] - starting_points['ID'][1]-10
    #     x = starting_points['ID'][0]+5
    #     y = starting_points['ID'][1]+5
    #     cell = img[y:y+h, math.floor(x+((i)*(w/8))):math.floor(x+((i+1)*(w/8)))]
    #     cv2.imwrite(path + "/" + "ID" + str(i) + ".jpg", cell)

    # # Student First Name extraction
    # for i in range(8):
    #     w = ending_points['First Name'][0] - starting_points['First Name'][0]-10
    #     h = ending_points['First Name'][1] - starting_points['First Name'][1]-10
    #     x = starting_points['First Name'][0]+5
    #     y = starting_points['First Name'][1]+5
    #     cell = img[y:y+h, math.floor(x+((i)*(w/8))):math.floor(x+((i+1)*(w/8)))]
    #     cv2.imwrite(path + "/" + "FN" + str(7-i) + ".jpg", cell)

    # # Student Last Name extraction
    # for i in range(8):
    #     w = ending_points['Last Name'][0] - starting_points['Last Name'][0]-10
    #     h = ending_points['Last Name'][1] - starting_points['Last Name'][1]-10
    #     x = starting_points['Last Name'][0]+5
    #     y = starting_points['Last Name'][1]+5
    #     cell = img[y:y+h, math.floor(x+((i)*(w/8))):math.floor(x+((i+1)*(w/8)))]
    #     cv2.imwrite(path + "/" + "LN" + str(7-i) + ".jpg", cell)

    # # BS extraction
    # cell = img[degree_points['BS'][1] : degree_points['BS'][1] + 25, degree_points['BS'][0] : degree_points['BS'][0] + 25]
    # cv2.imwrite(path + "/" + "BS.jpg", cell)

    # # MS extraction
    # cell = img[degree_points['MS'][1] : degree_points['MS'][1] + 25, degree_points['MS'][0] : degree_points['MS'][0] + 25]
    # cv2.imwrite(path + "/" + "MS.jpg", cell)

    # # PHD extraction
    # cell = img[degree_points['PHD'][1] : degree_points['PHD'][1] + 25, degree_points['PHD'][0] : degree_points['PHD'][0] + 25]
    # cv2.imwrite(path + "/" + "PHD.jpg", cell)


def calculate_distance(p1, p2):
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist


def extract_and_save(test_form_path, extracted_path, form_width, form_height, cell_width, cell_height):
    for image_path in glob.glob(test_form_path + "/*.*"):
        image = cv2.imread('data/test_form/forms/test.jpg')
        corners = np.array(aruco_extraction(image), dtype=np.float32)
        if corners is None:
            print(
                f"The image {image_path[image_path.find('/' + type + '/')+3:]} is dropped."
            )
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
        "test_form.test_form_path",
        "test_form.extracted_path",
        "pre_processing.form_width",
        "pre_processing.form_height",
        "pre_processing.cell_width",
        "pre_processing.cell_height"
    ]

    check_config_keys(config, required_keys)

    test_form_path = config["test_form"].get("test_form_path")
    extracted_path = config["test_form"].get("extracted_path")

    form_width = config["pre_processing"].get("form_width")
    form_height = config["pre_processing"].get("form_height")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    if not os.path.exists(extracted_path):
        os.makedirs(extracted_path)
        print(f"Folder '{extracted_path}' created.")
    else:
        print(f"Folder '{extracted_path}' already exists.")

    extract_and_save(test_form_path, extracted_path, form_width, form_height, cell_width, cell_height)


if __name__ == '__main__':
    config_path = "config/config.yaml"
    config = check_config_file(config_path)
    data_extraction(config)
