import numpy as np
import cv2
import math
import os


def find_main_corners(frame):
    # Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()
    # Detect the markers in the image
    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
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
    return p1, p3, p2, p4


def apply_perspective(img, corners):
    # extract the form to a 800 * 800 picture and correct its perspective
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
    H = cv2.getPerspectiveTransform(points1, points2)
    output_size = (J.shape[1], J.shape[0])
    J = cv2.warpPerspective(img, H, output_size)
    cv2.imshow('Form', J)
    cv2.waitKey(0)
    return J


def extracting_data(img):
    # extract each letter, number and checkbox
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
    # cv2.imshow('Corner', M)
    # cv2.waitKey(0)
    # guess the location of the principal points
    starting_points = {'ID': (40, 200), 'First Name': (40, 310), 'Last Name': (40, 400)}
    ending_points = {'ID': (585, 280), 'First Name': (585, 372), 'Last Name': (585, 470)}
    degree_points = {'PHD': (75, 510), 'MS': (220, 510), 'BS': (450, 510)}

    # extracting!
    # Student ID
    starting_flag = True
    ending_flag = True
    for i in range(800):
        if starting_flag or ending_flag:
            for j in range(800):
                if starting_flag or ending_flag:
                    if M[i][j] == 255:
                        if calculate_distance(starting_points['ID'], (j, i)) <= 13.0 and starting_flag:
                            starting_points['ID'] = (j, i)
                            # cv2.circle(img, (starting_points['ID']), 5, (0, 0, 255), -1)
                            starting_flag = False
                        if calculate_distance(ending_points['ID'], (j, i)) <= 13.0 and ending_flag:
                            ending_points['ID'] = (j, i)
                            # cv2.circle(img, (ending_points['ID']), 5, (0, 0, 255), -1)
                            ending_flag = False

    # Student First Name
    starting_flag = True
    ending_flag = True
    for i in range(800):
        if starting_flag or ending_flag:
            for j in range(800):
                if starting_flag or ending_flag:
                    if M[i][j] == 255:
                        if calculate_distance(starting_points['First Name'], (j, i)) <= 13.0 and starting_flag:
                            starting_points['First Name'] = (j, i)
                            # cv2.circle(img, (starting_points['First Name']), 5, (0, 0, 255), -1)
                            starting_flag = False
                        if calculate_distance(ending_points['First Name'], (j, i)) <= 13.0 and ending_flag:
                            ending_points['First Name'] = (j, i)
                            # cv2.circle(img, (ending_points['First Name']), 5, (0, 0, 255), -1)
                            ending_flag = False

    # Student Last Name
    starting_flag = True
    ending_flag = True
    for i in range(800):
        if starting_flag or ending_flag:
            for j in range(800):
                if starting_flag or ending_flag:
                    if M[i][j] == 255:
                        if calculate_distance(starting_points['Last Name'], (j, i)) <= 13.0 and starting_flag:
                            starting_points['Last Name'] = (j, i)
                            # cv2.circle(img, (starting_points['Last Name']), 5, (0, 0, 255), -1)
                            starting_flag = False
                        if calculate_distance(ending_points['Last Name'], (j, i)) <= 13.0 and ending_flag :
                            ending_points['Last Name'] = (j, i)
                            # cv2.circle(img, (ending_points['Last Name']), 5, (0, 0, 255), -1)
                            ending_flag = False

    # Student Degree BS
    point = degree_points['BS']
    flag = True
    for i in range(800):
        if flag:
            for j in range(800):
                if flag:
                    if M[i][j] == 255:
                        if calculate_distance(point, (j, i)) <= 10.0:
                            # cv2.circle(img, (j, i), 5, (255, 0, 0), -1)
                            degree_points['BS'] = (j, i)
                            flag = False
    # Student Degree MS
    point = degree_points['MS']
    flag = True
    for i in range(800):
        if flag:
            for j in range(800):
                if flag:
                    if M[i][j] == 255:
                        if calculate_distance(point, (j, i)) <= 15.0:
                            # cv2.circle(img, (j, i), 5, (255, 0, 0), -1)
                            degree_points['MS'] = (j, i)
                            flag = False
    # Student Degree PHD
    point = degree_points['PHD']
    flag = True
    for i in range(800):
        if flag:
            for j in range(800):
                if flag:
                    if M[i][j] == 255:
                        if calculate_distance(point, (j, i)) <= 15.0:
                            # cv2.circle(img, (j, i), 5, (255, 0, 0), -1)
                            degree_points['PHD'] = (j, i)
                            flag = False


    return starting_points, ending_points, degree_points


def save_data(img, starting_points, ending_points, degree_points, path):
    # save data in a folder
    os.mkdir(path)
    # Student ID extraction
    for i in range(8):
        w = ending_points['ID'][0] - starting_points['ID'][0]-10
        h = ending_points['ID'][1] - starting_points['ID'][1]-10
        x = starting_points['ID'][0]+5
        y = starting_points['ID'][1]+5
        cell = img[y:y+h, math.floor(x+((i)*(w/8))):math.floor(x+((i+1)*(w/8)))]
        cv2.imwrite(path + "/" + "ID" + str(i) + ".jpg", cell)
    
    # Student First Name extraction
    for i in range(8):
        w = ending_points['First Name'][0] - starting_points['First Name'][0]-10
        h = ending_points['First Name'][1] - starting_points['First Name'][1]-10
        x = starting_points['First Name'][0]+5
        y = starting_points['First Name'][1]+5
        cell = img[y:y+h, math.floor(x+((i)*(w/8))):math.floor(x+((i+1)*(w/8)))]
        cv2.imwrite(path + "/" + "FN" + str(7-i) + ".jpg", cell)
    
    # Student Last Name extraction
    for i in range(8):
        w = ending_points['Last Name'][0] - starting_points['Last Name'][0]-10
        h = ending_points['Last Name'][1] - starting_points['Last Name'][1]-10
        x = starting_points['Last Name'][0]+5
        y = starting_points['Last Name'][1]+5
        cell = img[y:y+h, math.floor(x+((i)*(w/8))):math.floor(x+((i+1)*(w/8)))]
        cv2.imwrite(path + "/" + "LN" + str(7-i) + ".jpg", cell)

    # BS extraction
    cell = img[degree_points['BS'][1] : degree_points['BS'][1] + 25, degree_points['BS'][0] : degree_points['BS'][0] + 25]
    cv2.imwrite(path + "/" + "BS.jpg", cell)

    # MS extraction
    cell = img[degree_points['MS'][1] : degree_points['MS'][1] + 25, degree_points['MS'][0] : degree_points['MS'][0] + 25]
    cv2.imwrite(path + "/" + "MS.jpg", cell)

    # PHD extraction
    cell = img[degree_points['PHD'][1] : degree_points['PHD'][1] + 25, degree_points['PHD'][0] : degree_points['PHD'][0] + 25]
    cv2.imwrite(path + "/" + "PHD.jpg", cell)


def calculate_distance(p1, p2):
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist


def main():
    addr = 'test.jpg'
    frame = cv2.imread(addr)
    corners = np.array(find_main_corners(frame), dtype=np.float32)
    form_image = apply_perspective(frame, corners)
    starting_points, ending_points, degree_points = extracting_data(form_image)
    save_data(form_image, starting_points, ending_points, degree_points, addr[0:-4])
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



