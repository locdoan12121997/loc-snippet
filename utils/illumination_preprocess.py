import cv2
import numpy as np


def image_show(name, image, resize=1):
    H, W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  # WINDOW_NORMAL
    # cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)  # WINDOW_GUI_EXPANDED
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize * W), round(resize * H))


def image_show_norm(name, image, min=None, max=None, resize=1):
    if max is None: max = image.max()
    if min is None: min = image.min()

    H, W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  # WINDOW_NORMAL
    cv2.imshow(name, ((image - min) / (max - min) * 255).astype(np.uint8))
    cv2.resizeWindow(name, round(resize * W), round(resize * H))


image = cv2.imread('run_data/test_images/elder_images.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
image = cv2.resize(image, (112, 112))

illumination = np.zeros((112, 112), np.float32)
# # center (0,0)= top,left 
# # circle(center, radius, color, thickness) # color la do manh hieu ung se apply vao anh
# cv2.circle(illumination, (112, 61), 50, 2, -1) #, lineType=cv2.LINE_AA
radius = int(np.random.uniform(40, 100))
center = tuple((np.random.rand(2) * 112).astype(int))
color = np.random.uniform(0.5, 1) * 2
cv2.circle(illumination, center, radius, color, -1)
# print('radius{}, color {}, center{}'.format(radius, color, center))
illumination = cv2.GaussianBlur(illumination, (257, 257), 0)
illumination = illumination.reshape(112, 112, 1)

augment = image.astype(np.float32) / 255
augment = augment * (1 + illumination)
augment = np.clip(augment * 255, 0, 255).astype(np.uint8)

image_show_norm('illumination', illumination)
image_show('image', image, )
image_show('augment', augment, )
cv2.waitKey()
