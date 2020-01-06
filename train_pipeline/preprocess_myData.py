"""
aioz.aiar.truongle - Sep 24, 2019
preprocess my data
"""
import cv2
import os
import Augmentor
import numpy as np
from tqdm import tqdm
from keras.utils import to_categorical

LIST_LABELS = ['cover', 'eat_drink', 'record', 'sleep', 'other']


def augment(ims_root, max_sample, size=(112, 112)):
    list_dir = os.listdir(ims_root)
    for dir_n in list_dir:
        data_dir = os.path.join(ims_root, dir_n)
        list_im = os.listdir(data_dir)
        p = Augmentor.Pipeline(data_dir)
        if max_sample // 3 < len(list_im) < max_sample:
            p.flip_left_right(probability=0.75)
        elif len(list_im) < max_sample // 3:
            p.flip_left_right(probability=0.3)
            p.shear(probability=0.5, max_shear_left=20, max_shear_right=20)
            p.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=20)
        else:
            p.resize(probability=1.0, width=size[0], height=size[1])
        p.sample(max_sample)


def save_npz(root, size=(224, 224)):
    list_lb = np.asarray(LIST_LABELS)
    list_dir = os.listdir(root)
    images = []
    labels = []
    print("[INFO] Process for save .npz ...")
    for dir_n in tqdm(list_dir):
        data_dir = os.path.join(root, dir_n)
        list_im = os.listdir(data_dir)
        for im_n in list_im:
            src = os.path.join(data_dir, im_n)
            im = cv2.imread(src)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            im = cv2.resize(im, size)
            images.append(im)
            labels.append(np.where(list_lb == dir_n)[0][0])
    images = np.asarray(images)
    labels = np.asarray(labels)
    print(labels)
    # convert label into one-hot vector
    labels = to_categorical(labels)

    # shuffle
    indices = np.random.permutation(len(labels))
    labels = labels[indices]
    images = images[indices]

    save_file = os.path.join(root, "data.npz")
    np.savez(save_file, images=images, labels=labels)
    print("[INFO] Save .npy at {}".format(save_file))
    print("[INFO] Number of images - labels: {} - {}".format(len(images), len(labels)))


def main():
    # TRAIN
    root = 'dataset/train'
    #augment(root, 350)
    save_npz(root)

    # VAL
    root = 'dataset/val'
    #augment(root, 20)
    save_npz(root)


if __name__ == '__main__':
    main()
