import numpy as np
import cv2

def crop_img_by_rect(self, img, rect, margin=0.0):
    x1 = rect.left()
    x2 = rect.right()
    y1 = rect.top()
    y2 = rect.bottom()
    # size of face
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    # add margin
    full_crop_x1 = x1 - int(w * margin)
    full_crop_y1 = y1 - int(h * margin)
    full_crop_x2 = x2 + int(w * margin)
    full_crop_y2 = y2 + int(h * margin)
    # size of face with margin
    new_size_w = full_crop_x2 - full_crop_x1 + 1
    new_size_h = full_crop_y2 - full_crop_y1 + 1

    # ensure that the region cropped from the original image with margin
    # doesn't go beyond the image size
    crop_x1 = max(full_crop_x1, 0)
    crop_y1 = max(full_crop_y1, 0)
    crop_x2 = min(full_crop_x2, img.shape[1] - 1)
    crop_y2 = min(full_crop_y2, img.shape[0] - 1)
    # size of the actual region being cropped from the original image
    crop_size_w = crop_x2 - crop_x1 + 1
    crop_size_h = crop_y2 - crop_y1 + 1

    # coordinates of region taken out of the original image in the new image
    new_location_x1 = crop_x1 - full_crop_x1
    new_location_y1 = crop_y1 - full_crop_y1
    new_location_x2 = crop_x1 - full_crop_x1 + crop_size_w - 1
    new_location_y2 = crop_y1 - full_crop_y1 + crop_size_h - 1

    new_img = np.random.randint(256, size=(new_size_h, new_size_w, img.shape[2])).astype('uint8')

    new_img[new_location_y1: new_location_y2 + 1, new_location_x1: new_location_x2 + 1, :] = \
        img[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1, :]

    # if margin goes beyond the size of the image, repeat last row of pixels
    if new_location_y1 > 0:
        new_img[0:new_location_y1, :, :] = np.tile(new_img[new_location_y1, :, :], (new_location_y1, 1, 1))

    if new_location_y2 < new_size_h - 1:
        new_img[new_location_y2 + 1:new_size_h, :, :] = np.tile(new_img[new_location_y2:new_location_y2 + 1, :, :],
                                                                (new_size_h - new_location_y2 - 1, 1, 1))
    if new_location_x1 > 0:
        new_img[:, 0:new_location_x1, :] = np.tile(new_img[:, new_location_x1:new_location_x1 + 1, :],
                                                   (1, new_location_x1, 1))
    if new_location_x2 < new_size_w - 1:
        new_img[:, new_location_x2 + 1:new_size_w, :] = np.tile(new_img[:, new_location_x2:new_location_x2 + 1, :],
                                                                (1, new_size_w - new_location_x2 - 1, 1))
    return new_img


def crop_bboxes(image, bboxes):
    '''
    crop multiple boxes in image
    '''
    imgs = []
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox[:4].astype(int)
        img = image[ymin:ymax, xmin:xmax]
        imgs.append(img)
    return imgs


def draw_bboxs(img, bboxes, color_bbox=(0, 255, 255)):
    """
    :param img: images need ro draw box
    :param bboxes: bounding box, format [xmin, ymin, xmax, ymax]
    :param color_bbox: color of bounding box
    :return: images has been drawn box
    """
    for i, bbox in enumerate(bboxes):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bbox, 2)

   
def draw_bboxs_with_label_action(image, bboxs, label_name):
    '''
    draw multiple bounding box with label
    '''
    color = (255, 0, 0)
    for i, bbox in enumerate(bboxs):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])    
        # # draw bbox
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # track id
        label = str(label_name[i])
        if label != 'nothing':
            # draw bbox
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image,
                            (xmin, ymin),
                            (xmin + 10 + text_size[0][0], ymin + 10 + text_size[0][1]),
                            color, -1)
            cv2.putText(image, label,
                        (xmin + 5, ymin + 5 + text_size[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
    