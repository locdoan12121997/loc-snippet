"""
aioz.aiar.truongle - Oct 05, 2019
un distort fish eye camera
"""
import cv2
import numpy as np


class UnDistortCamera:
    def __init__(self, width, height):
        # DIM [w, h] = (2304, 1296)
        DIM = (int(width), int(height))
        K = np.array(
            [[1501.2595834770557, 0.0, 1370.6697823250217], [0.0, 1500.79553264583, 788.6583790280063],
             [0.0, 0.0, 1.0]])
        D = np.array([[-0.1592635301272772], [0.19320207691352564], [-0.36744955632755505], [0.2280147761483244]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

    def run(self, frame):
        undistorted_img = cv2.remap(frame, self.map1, self.map2,
                                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img
