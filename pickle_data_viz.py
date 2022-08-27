import pickle

from cv2 import waitKey
import cv2
import numpy as np

with open('trjs.pickle', 'rb') as f:
    data = pickle.load(f)
    trajectories = data['trajectories']

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

frame = np.ones((400, 400, 3), np.uint8)*100

for trj in trajectories:
    color = COLORS[trj['color']]
    color = (int(color[0]), int(color[1]), int(color[2]))

    points = np.uint( trj['points']*1000 )
    cv2.drawContours(frame, [points], 0, color, 3)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
