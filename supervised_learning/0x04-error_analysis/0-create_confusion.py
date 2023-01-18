#!/usr/bin/env python3
"""task 0"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """function that creates a confusion matrix"""
    cm = np.dot(labels.T, logits)
