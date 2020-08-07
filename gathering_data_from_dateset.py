import cv2
import numpy as np
import glob
import math
import os


def aruco_extraction(img):
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()
    # Detect the markers in the image
    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
    assert len(marker_corners) == 4, "You have detected {} instead of 4 ArUco markers!".format(len(marker_corners))
    p1 = (0, 0)
    p2 = (0, 0)
    p3 = (0, 0)
    p4 = (0, 0)
    for m in range(len(marker_ids)):
        if marker_ids[m] == 30:
            p1 = marker_corners[m][0][0][0], marker_corners[m][0][0][1]
        elif marker_ids[m] == 33:
            p2 = marker_corners[m][0][2][0], marker_corners[m][0][2][1]
        elif marker_ids[m] == 31:
            p3 = marker_corners[m][0][1][0], marker_corners[m][0][1][1]
        else:
            p4 = marker_corners[m][0][3][0], marker_corners[m][0][3][1]
    return p1, p3, p2, p4


def apply_perspective(img, corners):
    # extract the form to a 800 * 800 picture and correct its perspective
    points1 = corners
    n = 800
    m = 1131
    J = np.zeros((m, n))
    points2 = np.array([(0, 0),
                        (n, 0),
                        (n, m),
                        (0, m)]).astype(np.float32)
    H = cv2.getPerspectiveTransform(points1, points2)
    output_size = (J.shape[1], J.shape[0])
    J = cv2.warpPerspective(img, H, output_size)
    return J


def extract_cells(img, path):
    # extract each letter, number and checkbox.
    # save them!

    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    E = cv2.Canny(G, 50, 100)

    kernel = np.ones((1,1),np.uint8)
    T = cv2.erode(E,kernel)

    G = np.float32(T)
    window_size = 4
    soble_kernel_size = 3
    alpha = 0.04
    H = cv2.cornerHarris(G, window_size, soble_kernel_size, alpha)
    H = H / H.max()
    M = np.uint8(H > 0.005) * 255

    starting_points = {'0': (115, 0), '1': (115, 50), '2': (0, 110), '3': (0, 163), '4': (0, 218), '5': (0, 270),
                       '6': (0, 322), '7': (0, 380), '8': (0, 435), '9': (0, 487), '10': (0, 542), '11': (0, 599),
                       '12': (0, 650), '13': (0, 705), '14': (0, 760), '15': (0, 815), '16': (0, 867), '17': (0, 922),
                       '18': (0, 977), '19': (115, 1030), '20': (115, 1083)}
    ending_points = {'0': (687, 50), '1': (687, 105), '2': (800, 160), '3': (800, 215), '4': (800, 270),
                     '5': (800, 325), '6': (800, 378), '7': (800, 433), '8': (800, 488), '9': (800, 543),
                     '10': (800, 598), '11': (800, 653), '12': (800, 708), '13': (800, 763), '14': (800, 815),
                     '15': (800, 868), '16': (800, 923), '17': (800, 978), '18': (800, 1028), '19': (687, 1085),
                     '20': (687, 1131)}

    directory = '-1'
    for row in range(21):
        if path[-6] == '1':
            if row < 2:
                directory = str(row)
            elif row > 18:
                directory = str(row-17)
            else:
                directory = str(row+8)
        elif path[-6] == '2':
            if row < 2:
                directory = str(row+4)
            elif row > 16:
                directory = str(row-11)
            else:
                directory = str(row+25)

        column = 14
        if row < 2 or row > 18:
            column = 10
        for i in range(column):
            w = ending_points[str(row)][0] - starting_points[str(row)][0]-14
            h = ending_points[str(row)][1] - starting_points[str(row)][1]-14
            x = starting_points[str(row)][0]+7
            y = starting_points[str(row)][1]+7
            cell = img[y:y+h, math.floor(x+(i*(w/column))):math.floor(x+((i+1)*(w/column)))]
            cell = cv2.resize(cell, (60, 60))
            cell = cell[10:50, 10:50]
            cv2.imwrite("extracted_dataset" + "/" + directory + "/" + path[-14:-4] + "_"
                        + str(i) + ".jpg", cell)


def make_directories():
    path = "extracted_dataset"
    os.mkdir(path)
    for i in range(42):
        os.mkdir(path + '/' + str(i))


def calculate_distance(p1, p2):
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist
        

def main():
    flag = True
    make_directories()
    # you should upload the dataset file!
    for path in glob.glob('dataset' + '/*.*'):
        print(path)
        image = cv2.imread(path)
        corners = np.array(aruco_extraction(image), dtype=np.float32)
        form_image = apply_perspective(image, corners)
        extract_cells(form_image, path)
            

if __name__ == '__main__':
    main()
