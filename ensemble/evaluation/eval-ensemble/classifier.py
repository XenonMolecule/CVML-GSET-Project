import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
import cv2
import configparser as cp
import json
from math import ceil
from prediction import Class_Prediction
config = cp.RawConfigParser()
config.read('C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\classifier-dataset\\config\\config.cfg')
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
    factor = 1
    if((cur_height > 480) or (cur_width > 640)):
        if(cur_height * (4/3) > cur_width):
            factor = 480 / cur_height
        else:
            factor = 640 / cur_width
        gray_image = cv2.resize(gray_image, (0,0), fx=factor, fy=factor)
        cur_height, cur_width = gray_image.shape
    for d in dimensions:
        if cur_height <= d[0] and cur_width <= d[1]:
            new_height = d[0]
            new_width = d[1]
            break
    border_width = (new_width - cur_width) // 2
    border_height = (new_height - cur_height) // 2
    rgb = [255, 255, 255]
    border_image = cv2.copyMakeBorder(gray_image, border_height, border_height, border_width, border_width, cv2.BORDER_CONSTANT, value = rgb)
    resized_image = cv2.resize(border_image, (160, 120))
    return resized_image

def predict_img(image, display):
    image = preprocess(image)
    fd = hog(image, orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block, visualize = False)
    clf = joblib.load(model_linear_path)
    pred = clf.predict([fd])[0]
    prob = clf.predict_proba([fd])[0][pred - 1]
    return Class_Prediction(prob, pred)
