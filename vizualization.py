import numpy as np
import pandas as pd
import os
import traceback
import time
import matplotlib.pyplot as plt
import sys
import pickle
import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


colors = [
    [ 35,  57,  40], # G762
    [252, 253, 254], # nothing
    [ 80, 151, 217], # 24
    [119, 193, 228], # 36
    [ 67, 105,  66], # 55
    [ 50,  80,  42], # G554
    [ 21,  30,  17], # B962
    [ 78, 132, 170], # 101
    [101, 150, 115], # G606
    [ 49,  83, 139], # 91
    [175, 216, 223], # Y000
    [221, 236, 242]  # WG1
]

# directory = '/home/human/DEMONSTRATION/Drawing/Segmentation/Trj'
filename = './trjs.pickle' 
img = cv2.imread('./branch.jpg', cv2.IMREAD_UNCHANGED)

with open('trjs.pickle', 'rb') as file:
    data = pickle.load(file)

trjs = data['trajectories']
img_contours_global = np.zeros(img.shape, np.uint8)

for trj in trjs:
    cnt = []
    for p in trj['points']:
        # print(p)
        coordinates = p['p'] 
        coordinates[0] *= 1000
        coordinates[1] *= 1000
        color = p['color']
        # print(colors[color])
        cnt.append([coordinates])
        # print(cnt)
    # print(cnt)
    cnt = np.array(cnt, dtype=np.int32)
    
    cv2.drawContours(img_contours_global, [cnt], -1, colors[color], 3)
    # cv2.imshow('parent contour', img_contours_global)
    # cv2.waitKey(10)

cv2.imshow('parent contour', img_contours_global)
cv2.waitKey(0)
# X = data[:, 0]
# Y = data[:, 1]
# plt.plot(X, Y)
# plt.show()

# [ [[x,y]], [[x,y]],[[x,y]],[[x,y]],[[x,y]],]
