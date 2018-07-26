import numpy as np

class Box(object):
    def __init__(self, y1, x1, y2, x2, confidence, class_label):
        self.y1 = y1
        self.x1 = x1
        self.y2 = y2
        self.x2 = x2
        self.confidence = confidence
        self.class_label = class_label

    def __repr__(self):
        return str(self.class_label) + "(" + str(self.get_confidence()) + ") : " + str(self.get_coords())

    # slice up the image to the box size
    def splice_img(self, image):
        im_height = image.shape[0]
        im_width = image.shape[1]
        return (image[int(self.y1*im_height):int(self.y2*im_height), int(self.x1*im_width):int(self.x2*im_width)])

    # Get absolute coordinates
    def get_coordinates_absolute(self, image):
        im_height = image.shape[0]
        im_width = image.shape[1]
        return [int(self.y1*im_height), int(self.x1*im_width), int(self.y2*im_height), int(self.x2*im_width)]

    # Get relative coordinates
    def get_coords(self):
        return [self.y1, self.x1, self.y2, self.x2]

    # Get confidence score
    def get_confidence(self):
        return self.confidence

    # Get class label
    def get_class_label(self):
        return self.class_label

    # Get box based on center coordinates and such craziness
    def conv_to_center_mode(self):
        box_repr = []
        coords = self.get_coords() # relative coordinates
        box_repr.append((coords[1] + coords[3])/2) # center x       | 0
        box_repr.append((coords[0] + coords[2])/2) # center y       | 1
        box_repr.append(coords[3]-coords[1]) # width                | 2
        box_repr.append(coords[2]-coords[0]) # height               | 3
        box_repr.append(self.get_class_label()) # class label       | 4
        box_repr.append(self.get_confidence()) # confidence scores  | 5
        return box_repr

    def create_from_center_mode(self, center_box):
        self.y1 = (center_box[1] - center_box[3]/2)
        self.x1 = (center_box[0] - center_box[2]/2)
        self.y2 = (center_box[1] + center_box[3]/2)
        self.x2 = (center_box[0] + center_box[2]/2)
        self.class_label = center_box[4]
        self.confidence = center_box[5]
        return self

class Prediction(object):
    def __init__(self):
        self.prediction_boxes = []

    def append_box(self, box):
        self.prediction_boxes.append(box)

    def get_boxes(self):
        return self.prediction_boxes

    def get_coords(self):
        coords = []
        for box in self.prediction_boxes:
            coords.append(box.get_coords())
        return coords

    def get_confidences(self):
        confidences = []
        for box in self.prediction_boxes:
            confidences.append(box.get_confidence())
        return confidences

    def get_class_labels(self):
        labels = []
        for box in self.prediction_boxes:
            labels.append(box.get_class_label())
        return labels

    def get_ymins(self):
        ymins = []
        for box in self.prediction_boxes:
            ymins.append(box.get_coords[0])
        return ymins

    def get_xmins(self):
        xmins = []
        for box in self.prediction_boxes:
            xmins.append(box.get_coords[1])
        return xmins

    def get_ymaxs(self):
        ymaxs = []
        for box in self.prediction_boxes:
            ymaxs.append(box.get_coords[2])
        return ymins

    def get_xmaxs(self):
        xmaxs = []
        for box in self.prediction_boxes:
            xmaxs.append(box.get_coords[3])
        return xmaxs

class Class_Prediction(object):
    def __init__(self, confidence, class_label):
        self.confidence = confidence
        self.class_label = class_label

    def get_confidence(self):
        return self.confidence
