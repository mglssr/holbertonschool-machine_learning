#!/usr/bin/env python3
"""Task 0"""
import tensorflow.keras as K


class Yolo():
    """ uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class constructor"""
        self.model = K.models.load_model(model_path)
        f = open(classes_path, "r")
        self.class_names = f.read().split()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
