"""
aioz.aiar.truongle - Sep 24, 2019
train action
labels
'{
0-use phone
1-drink
2-eat
3-cover face
4-talking
5-sleep
6-record pen
7-record phone
}
'
"""
import os
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from termcolor import colored
import efficientnet.tfkeras as efficientnet
from helpers.generator import My_Generator
from helpers import helpers
from helpers.save_history import SaveHistory
from tensorflow.python.keras import models, layers, optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--method', default="effB1_fcl_affineScale", type=str, help='over view about method')
parser.add_argument('--data_train', default='dataset/train/data.npz')
parser.add_argument('--data_val', default='dataset/val/data.npz')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--model_dir', type=str, default='saved_models/effB1_fcl_affineScale_test')
parser.add_argument('--pretrained_model', default=None, type=str)
parser.add_argument('--num_classes', default=5, type=int)
parser.add_argument('--focal_loss', default=False, type=bool, help='if True use focal loss, else use CE')
args = parser.parse_args()


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def focal_loss_fixed(y_true, y_pred):
    """
    ref: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66146
    """
    gamma = 2.
    alpha = .25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def vis(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    f1 = history.history['f1']
    val_f1 = history.history['val_f1']

    epochs = range(len(acc))
    plt.style.use("seaborn")
    plt.plot(epochs, acc, 'green', label='Training acc')
    plt.plot(epochs, val_acc, 'red', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(args.model_dir, "vis_accuracy.png"))

    plt.figure()
    plt.plot(epochs, loss, 'green', label='Training loss')
    plt.plot(epochs, val_loss, 'red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(args.model_dir, "vis_loss.png"))

    plt.figure()
    plt.plot(epochs, f1, 'green', label='Training F1')
    plt.plot(epochs, val_f1, 'red', label='Validation F1')
    plt.title('Training and validation F1')
    plt.legend()
    plt.savefig(os.path.join(args.model_dir, "vis_f1.png"))


def train():
    helpers.check_dir(args.model_dir)

    #  WRITE ARGS
    args_file = os.path.join(args.model_dir, "commandline_args.txt")
    with open(args_file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(colored("[INFO] Write commandline_args  at </ {} />".format(args_file), "yellow", attrs=['bold']))

    # LOAD DATA
    print(colored("[INFO] Load data ... ", "cyan", attrs=['bold']))
    print("[INFO] Load train at </ {} />".format(args.data_train))
    data_train = np.load(args.data_train)
    x_train, y_train = data_train['images'], data_train['labels']

    print("[INFO] Load val at </ {} />".format(args.data_val))
    data_val = np.load(args.data_val)
    x_test, y_test = data_val['images'], data_val['labels']
    print("[INFO] Load data is done")

    # LOAD PRE-TRAIN
    if args.pretrained_model is not None:
        print(
            colored("[INFO] Load pre-trained model at </ {} />\n".format(args.pretrained_model), "red", attrs=['bold']))

        def swish(x):
            return tf.nn.swish(x)

        model = models.load_model(args.pretrained_model, custom_objects={'focal_loss_fixed': focal_loss_fixed,
                                                                          'swish': swish,
                                                                          'f1': f1})

    else:
        def dice_coef_loss(y_true, y_pred):
            return y_true

        model_segment = tf.keras.models.load_model('./segmentation_action/saved_models/model.h5',
                           custom_objects={'dice_coef_loss': dice_coef_loss,
                                           'dice_coef_metric': dice_coef_loss}, compile=False)
        inputs = tf.keras.layers.Input((224, 224, 3))

        masks = model_segment(inputs) # [B, 224, 224, 1]

        # freeze segment branch
        # for layer in model_segment.layers:
        #     layer.trainable = False

        input_cls = tf.keras.layers.Multiply()([masks, inputs])
        base_model = efficientnet.EfficientNetB1(weights='imagenet', include_top=False)
        x = base_model(input_cls)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(args.num_classes, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=predictions)
        #model.load_weights('./saved_models/effB1_fcl_affineScale_test/myData_effB1_fcl_affineScale_model.003.h5')

    # summary model
    model.summary()

    # focal loss
    if args.focal_loss:
        print("[INFO] Use focal loss ... ")
        loss = focal_loss_fixed
    else:
        print("[INFO] Use CE loss ... ")
        loss = 'categorical_crossentropy'

    # save
    save_dir = os.path.join(os.getcwd(), args.model_dir)
    model_name = 'myData_%s_model.{epoch:03d}.h5' % args.method
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, model_name)

    checkpoint = ModelCheckpoint(filepath=file_path,
                                 monitor='val_f1',
                                 verbose=1,  # show val_Acc improved in print
                                 save_best_only=False, mode='max')
    # SaveHistory: save dictionary training into pickle, information include: loss, acc, ...
    callbacks = [checkpoint, SaveHistory(save_dir=args.model_dir)]

    train_datagen = My_Generator(x_train, y_train, batch_size=args.batch_size, is_train=True, augment=True)
    model.compile(loss=loss,
                  optimizer=optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy', f1])

    history = model.fit_generator(train_datagen,
                                  epochs=args.epoch,
                                  validation_data=(x_test, y_test),
                                  callbacks=callbacks,
                                  class_weight=train_datagen.class_weight,
                                  shuffle=True)  # Note: shuffle only work with generator implements keras.utils.Sequence. If not shuffle is depeneded on generator

    # VISUALIZE
    vis(history)


if __name__ == '__main__':
    train()
