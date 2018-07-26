import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
import cv2
import configparser as cp
import json
from math import ceil
from prediction import Class_Prediction
config = cp.RawConfigParser()
config.read('../../../classifier-dataset/config/config.cfg')
orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
normalize = config.getboolean("hog", "normalize")
model_linear_path = config.get("paths", "model_linear_path")

dimensions = [(3 * i, 4 * i) for i in range(40, 161)]

def preprocess(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cur_height, cur_width = gray_image.shape
    new_height, new_width = 480, 640
    for d in dimensions:
        if cur_height <= d[0] and cur_width <= d[1]:
            new_height = d[0]
            new_width = d[1]
            break
    border_width = (new_width - cur_width) // 2
    border_height = (new_height - cur_height) // 2
    rgb = [255, 255, 255]
    border_image = cv2.copyMakeBorder(gray_image, border_height, border_height, border_width, border_width, cv2.BORDER_CONSTANT, value = rgb)
    cv2.imshow("border_image", border_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    resized_image = cv2.resize(border_image, (160, 120))
    cv2.imshow("resized_image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return resized_image

def predict_img(image):
    image = preprocess(image)
    fd = hog(image, orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block, visualize = False)
    clf = joblib.load(model_linear_path)
    print(clf.classes_)
    print(clf.predict_proba([fd]))
    pred = clf.predict([fd])[0]
    prob = clf.predict_proba([fd])[pred - 1]
    return Class_Prediction(prob, pred)

print(predict_img(cv2.imread("test3.jpg")))