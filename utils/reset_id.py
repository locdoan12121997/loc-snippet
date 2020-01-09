import cv2
import os
import glob
import shutil


root_dir = "temp"
n_class = ['abnormal', 'normal']


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def copy_all_data():
    copy_to = "id_reset"

    for scene_class in get_immediate_subdirectories('temp'):
        k=0
        if os.path.isdir(copy_to + "/" + scene_class) == False:
            os.mkdir(copy_to + "/" + scene_class)


        files = [f for f in glob.glob(root_dir + "/" + scene_class + "/*.jpg")]
        for i, f in enumerate(files):
            shutil.copy2(f, copy_to + "/" + scene_class + "/" + str(k) + ".jpg")
            k += 1


copy_all_data()
