import random

import cv2
import numpy as np


random.seed(0)
n_classes = 32
class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]

for i in range(n_classes):
    img = class_colors[i] * np.ones((720, 480, 3))
    cv2.imwrite("temp/%d.png" %(i) , img)

