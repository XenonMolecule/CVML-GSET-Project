import numpy as np

from prediction import Box
from prediction import Prediction

def predict_img(image, display):
    prediction = Prediction()
    prediction.append_box(Box(0.1,0.1,0.9,0.9,0.5,1))
    return prediction
