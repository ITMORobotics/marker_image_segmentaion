from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import numpy as np
import cv2
import scipy.spatial as sp


def plot_palette(colors, figname = 'colors'):
    tmp = np.tile(colors[0], (100, 100, 1))

    for i in range(1, len(colors)):
        tmp = np.concatenate((tmp, np.tile(colors[i], (100, 100, 1))), axis=1)

    cv2.imshow(figname, tmp)

def find_nearest_color(colors):
    global main_colors

    tree = sp.KDTree(main_colors)

    idx = []
    for c in colors:
        data = tuple(c)
        ditsance, result = tree.query(data)
        idx.append(result)

    return idx

def generate_colors_order(colors):
    colors_hsv = -cv2.cvtColor(np.array([colors], np.uint8), cv2.COLOR_RGB2HSV)[0]
    return colors_hsv[:, 2].argsort()


if __name__ == '__main__':

    img = cv2.imread('lesha.png')

    l = 960
    vshift = 30
    startv = img.shape[0]//2 - l//2 + vshift
    endv = img.shape[0]//2 + l//2 + vshift
    starth = img.shape[1]//2 - l//2
    endh = img.shape[1]//2 + l//2

    img = img[startv:endv:, starth:endh]
    img = cv2.resize(img, [400, 400])




    cv2.imshow('img', img)

    labels1 = segmentation.slic(img, compactness=40, n_segments=500, start_label=1)

    img2 = np.ones(img.shape)*255
    out2 = color.label2rgb(labels1, img2, kind = 'avg', bg_label = 0)

    out1 = color.label2rgb(labels1,  img, kind = 'avg', bg_label = 0)
    out1 = out1.astype(np.uint8)

    img_with_branch = segmentation.mark_boundaries(img2, labels1)
    img_with_branch_2 = segmentation.mark_boundaries(out1, labels1)
    img_with_branch_3 = segmentation.mark_boundaries(out2, labels1)
    img_with_branch_3 = img_with_branch_3.astype(np.uint8)




    Z = out1.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 12
    ret,label,center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((out1.shape))


    idx = generate_colors_order(center)
    plot_palette(center[idx], 'centers')

    print('center\n', center)
    print('idx\n', idx)

    cv2.imshow('res', res2)

    cv2.imwrite('./mask.jpg', res2)
    cv2.imwrite('./branch.jpg', img_with_branch)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
