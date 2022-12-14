from cv2 import imshow
import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import sys
import os
import roboticstoolbox.tools.trajectory as rtb
import time
import pickle

sys.path.append(os.path.dirname(os.path.dirname(__file__)))



img = cv2.imread('./branch.jpg', cv2.IMREAD_UNCHANGED)
source_image = cv2.imread('./mask.jpg', cv2.IMREAD_UNCHANGED)

kernel = np.ones((5, 5), 'uint8')
img_contours_global = np.zeros(img.shape, np.uint8)
filled_contour = np.array([[]])
binary_mask = np.zeros(img.shape[0:2])

COLORS = np.array([
    [247, 250, 251],
    [ 34,  54,  26],
    [ 72, 137, 212],
    [ 63, 101,  62],
    [111, 189, 234],
    [ 61, 107, 153],
    [ 45,  74,  37],
    [102, 146, 111],
    [ 19,  28,  17],
    [196, 229, 236],
    [ 34,  49,  75],
    [158, 203, 209]
], np.uint8)

COLORS_ORDER_IDX = [9, 4, 2, 11, 5, 7, 3, 10, 6, 1, 8, 0]

data = {
    'trajectories':[]
}

count = 0
POHIBITED_AREA = 1

def reshape_contour(cnt):
    trajectory = cnt
    trajectory = np.reshape(trajectory, (len(trajectory), -1))
    trajectory = np.array(trajectory)
    trajectory = trajectory.astype(np.float64)

    return trajectory

def check_area(cnt):
    global POHIBITED_AREA

    return 	cv2.contourArea(cnt) > POHIBITED_AREA

def check_intersect(cnt):
    global binary_mask
    global filled_contour
    cnt = reshape_contour(cnt)
    for point in cnt:
        if binary_mask[int(point[0])][int(point[1])] == 0:
            return True
    return False

def add_to_mask(cnt, flag):  # contour which will be added; if flag == 1 will clear binary mask
    global binary_mask

    if flag == 1:
        binary_mask = np.zeros(img.shape[0:2])

    for point in cnt:
        binary_mask[int(point[0])][int(point[1])] = 1

    return



def save_to_file(cnt):
    global count
    global current_color
    global COLORS
    global data
    global source_image

    x = int(cnt[0][0])
    y = int(cnt[0][1])

    current_color = source_image[y][x]
    current_color = [int (current_color[0]), int (current_color[1]), int (current_color[2])]

    label_color = find_nearest_color(current_color)



    trajectory = cnt.copy()
    trajectory[:, 1] = 400-trajectory[:, 1]
    trajectory /= 1000.0

    trj_array = rtb.mstraj(trajectory, dt=0.002, qdmax=0.25, tacc=0.05)
    trj = {'points': trj_array.q, 'width': 1.0, 'color': label_color}
    data['trajectories'].append(trj)




def depth_search(index, hierarchy, output_array):  #start point index; hierarchy of contours; the array of childs indexes
    if hierarchy[index][3] == -1:
        child_index = hierarchy[index][2]
    else:
        child_index = hierarchy[index][0]

    if child_index == -1:
        return output_array

    if (output_array.shape[0] == 0):
        output_array = np.array([child_index])
    else:
        output_array = np.append(output_array, child_index)

    output_array = depth_search(child_index, hierarchy, output_array)
    return output_array


def get_contour(img, flag): # if flag is 0 will return all contours else will return internal contours
    #convert img to grey
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #set a thresh
    thresh = 10
    #get threshold image
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if flag == 0:
        return contours

    if len(contours) == 0:
        return []

    contour_index = hierarchy[0]
    for i, index in enumerate(contour_index):
        cnt = contours[i]
        fill_contour(cnt, img)

def get_color(cnt):
    global source_image
    global current_color

    x = cnt[0][0][0]
    y = cnt[0][0][1]

    current_color = source_image[y][x]
    color = (int (current_color[0]), int (current_color[1]), int (current_color[2]))
    # color = list(np.random.random(size=3) * 256)
    return color


def find_nearest_color(color):
    global COLORS

    diff = np.linalg.norm(np.array(color) - COLORS, axis=1)
    print(diff[np.argmin(diff)])
    return np.argmin(diff)


def find_nearest_point(control_point, cnt):
    min = 100000
    index = -1
    for i, point in enumerate(cnt):
        error = np.linalg.norm(control_point - point, 2)
        if min > error:
            index = i
            min = error

    if min >= 10:
        return -1

    return index

def fill_contour(cnt, IMG):
    global img_contours_global
    global filled_contour
    global COLORS

    if check_area(cnt) == False:
        return

    color = COLORS[find_nearest_color(get_color(cnt))]
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.drawContours(img_contours_global, [cnt], 0, color, 3)

    cnt = reshape_contour(cnt)

    if filled_contour.shape[1] != 0:
        last_point = filled_contour[-1]
        index = find_nearest_point(last_point, cnt)
        if index == -1:
            save_to_file(filled_contour)
            filled_contour = cnt
            # add_to_mask(cnt, 1)
        else:
            cnt = np.roll(cnt, -index, axis=0)
            filled_contour = np.concatenate((filled_contour, cnt), axis=0)
            # add_to_mask(cnt, 0)
    else:
        filled_contour = cnt

    cv2.imshow('parent contour', img_contours_global)
    cv2.waitKey(2)

    # kernel = np.ones((3, 3), np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(IMG, kernel, iterations = 1)

    contours = get_contour(erosion, 1)

    return


# Main part
if __name__ == '__main__':

    current_color = [0, 0, 0]

    contours = get_contour(img, 0)

    for i in range(len(contours)):
        filled_contour = np.array([[]])
        if (len(contours[i]) <= 1):
            continue

        img_with_contour = np.zeros(img.shape, np.uint8)
        cv2.drawContours(img_with_contour, [contours[i]], 0, get_color(contours[i]), thickness=cv2.FILLED)

        # kernel = np.ones((3, 3), np.uint8) 1 attempt
        # kernel = np.ones((4, 4), np.uint8) 2 attempt
        kernel = np.ones((4, 4), np.uint8)
        dilation = cv2.dilate(img_with_contour, kernel, iterations = 1)

        fill_contour(contours[i], dilation)
        if (len(filled_contour) <= 1):
            continue
        save_to_file(filled_contour)

    # Sort data by color
    data['trajectories'].sort(key = lambda x: COLORS_ORDER_IDX.index( x['color'] ))
    with open('./trjs.pickle', 'wb') as file:
        pickle.dump(data, file)

    cv2.destroyAllWindows()
