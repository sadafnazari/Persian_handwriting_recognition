import glob
import os

import cv2
import numpy as np


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
                + img_path[img_path.find("/" + type + "/") + 3: -4]
                + "_"
                + str(col)
                + ".jpg",
                cell,
            )


def make_directories(extracted_dataset_path):
    """
    Creates the main directory for storing the result
    of data extraction and the needed subdirectories.
    0-9 are correspondent to numbers and 10-41 are
    correspondent to the persian letters.

    Args:
        extracted_dataset_path (str): The name of the main directory for
        storing the extracted dataset
    """
    # Creating the main directory
    if not os.path.exists(extracted_dataset_path):
        os.makedirs(extracted_dataset_path)
        print(f"Folder '{extracted_dataset_path}' created.")
    else:
        print(f"Folder '{extracted_dataset_path}' already exists.")

    # Creating the subdirectories within the main directory
    for i in range(42):
        if not os.path.exists(extracted_dataset_path + "/" + str(i)):
            os.makedirs(extracted_dataset_path + "/" + str(i))


def main():

    form_width = 800
    form_height = 1130

    cell_width = 60
    cell_height = 60

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    dataset_path = os.path.join(data_dir, "02_splitted")

    extracted_dataset_path = os.path.join(data_dir, "03_labeled")

    make_directories(extracted_dataset_path)

    # Extracting the forms of the type 'a'
    for image_path in glob.glob(dataset_path + "/a" + "/*.*"):
        image = cv2.imread(image_path)
        corners = aruco_extraction(image)
        if corners is None:
            print(f"The image {image_path[image_path.find('/a/')+3:]} is dropped.")
            continue
        form = form_extraction(image, corners, form_width, form_height)
        cell_extraction(
            form,
            image_path,
            extracted_dataset_path,
            "a",
            form_width,
            form_height,
            cell_width,
            cell_height,
        )

    # Extracting the forms of the type 'b'
    for image_path in glob.glob(dataset_path + "/b" + "/*.*"):
        image = cv2.imread(image_path)
        corners = aruco_extraction(image)
        if corners is None:
            print(f"The image {image_path[image_path.find('/b/')+3:]} is dropped.")
            continue
        form = form_extraction(image, corners, form_width, form_height)
        cell_extraction(
            form,
            image_path,
            extracted_dataset_path,
            "b",
            form_width,
            form_height,
            cell_width,
            cell_height,
        )

    print("dataset is extracted successfully.")


if __name__ == "__main__":
    main()
