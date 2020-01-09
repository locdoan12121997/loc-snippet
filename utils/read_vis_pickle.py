"""
aioz.aiar.truongle - Oct 10, 2019
read, visualize pickle
"""
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

parse = argparse.ArgumentParser()
parse.add_argument('--pickle_file', default="/home/aioz-truong/MonData/Project/aiar_AICX_project/src/gender_estimation/saved_model/effB4_imdb_fcl_multiScale/history.pickle",
                   type=str, help='pickle path for read and visualize')
arg = parse.parse_args()


def vis(data):
    acc = data['acc']
    val_acc = data['val_acc']
    loss = data['loss']
    val_loss = data['val_loss']

    epochs = range(len(acc))
    plt.style.use("seaborn")
    plt.plot(epochs, acc, 'green', label='Training acc')
    plt.plot(epochs, val_acc, 'red', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'green', label='Training loss')
    plt.plot(epochs, val_loss, 'red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def main():
    file = open(arg.pickle_file, 'rb')
    data = pickle.load(file)
    print(data.keys())
    vis(data)


if __name__ == '__main__':
    main()
