import numpy as np
import cv2
import pandas as pd
import os

def read_images(path, start_index = 0, end_index = -1):
    colour_images = []
    index = start_index
    for file in os.listdir(path):

        if index == end_index:
            break
        else:
            index += 1

        img = cv2.imread(path+file)
        img = cv2.resize(img, dsize=(299, 299))
        colour_images.append(img)

    colour_images = np.array(colour_images)
    return colour_images

def normalize_images(array):
    (length, height, width, channels) = array.shape
    array.astype(dtype=np.float32, copy=False)
    mean = array.sum(axis=0) / length
    array = np.subtract(array, mean, casting = 'same_kind')
    variance = (array ** 2).sum(axis=0) / length
    array = np.divide(array, variance, casting = 'same_kind')
    return array, mean, variance
