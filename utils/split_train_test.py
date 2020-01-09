import os
from random import shuffle
import numpy as np
import shutil
data_root = '/media/aioz-chien/C_database1/action_dataset/actions_data/final_data_actions'
train_prob = 0.8
# create training and testing folder
training_path = data_root + '/training'
testing_path = data_root + '/testing'
print(os.path.isdir(training_path))
if not os.path.isdir(training_path):
    os.mkdir(training_path)
if not os.path.isdir(testing_path):
    os.mkdir(testing_path)

folders = os.listdir(data_root)
folders.remove('testing')
folders.remove('training')
for folder in folders:
    images = os.listdir(os.path.join(data_root, folder))
    num_image = len(images)
    shuffle(images)
    # print(int(np.ceil(train_prob*num_image)))
    training_image = images[:int(np.ceil(train_prob*num_image))]
    testing_image = images[int(np.ceil(train_prob*num_image)):]
    # create sub_class folder
    if not os.path.isdir(os.path.join(training_path, folder)):
        os.mkdir(os.path.join(training_path, folder))
    if not os.path.isdir(os.path.join(testing_path, folder)):
        os.mkdir(os.path.join(testing_path, folder))  
    # Write training and testing image
    for image in training_image:
        shutil.copy(os.path.join(data_root, folder) + '/' + image, os.path.join(training_path, folder) +'/' + image)
    for image in testing_image:
        shutil.copy(os.path.join(data_root, folder) + '/' + image, os.path.join(testing_path, folder) + '/' + image)
