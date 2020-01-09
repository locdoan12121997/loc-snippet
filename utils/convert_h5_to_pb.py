'''
chien@aioz
Load check point keras = keras.models.load_model
define the custom object, name function if we use Lambda layer
then use tf.train.write_graph to save to pb file
For Age_estimation with efficientnetBO
h5: 50MB  pb: 16.6MB. Not prunning yet.
'''
import os
import sys
sys.path.append(os.getcwd())
# from keras.layers import Input, Average
# from keras.models import Model
from termcolor import colored

# age_estimation
# from keras.models import load_model
# from keras import backend as K
# # Action 
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

import tensorflow as tf
from tensorflow.python.framework import graph_util
from config import Config
import argparse
from tensorflow.python.keras.initializers import glorot_uniform

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def swish(x):
    return tf.nn.swish(x)


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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def write_model_architecture(model, file_arch):
    # write model architecture
    with open(file_arch, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def main():
    config = Config()
    K.set_learning_phase(0)  # make sure its testing mode
    sess = K.get_session()
    # Load keras model to find the output node name
    # model = load_fsanet_model(config=config)
    # print(config.Age_h5_checkpoint_name)

    print("[INFO] Load model from </ {} />".format(config.gender_h5_path))
    # Gender
    model = load_model(config.gender_h5_path, custom_objects={'swish': swish}, compile=False)
    # Emotion
    # model = load_model(config.expr_h5_path, custom_objects={'swish': swish}, compile=False)
    # Age estimation model
    # model = load_model(config.age_h5_checkpoint, custom_objects={'swish': swish, 'tf': tf}, compile=False)
    # Action model
    # model = load_model(args.checkpoint_dir,
    #                    custom_objects={'swish': swish, 'GlorotUniform': glorot_uniform, 'f1': f1, 'tf': tf},
    #                    compile=False)
    converted_output_node_names = [node.op.name for node in model.outputs]
    print("[INFO] in: ", model.inputs)
    print("[INFO] out: ", model.outputs)

    constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            converted_output_node_names)
    if not os.path.isdir(config.freeze_folder):
        os.mkdir("config.age_freeze_folder")
    name = os.path.split(config.gender_pb_path)[-1].split(".")[0]
    tf.io.write_graph(constant_graph, config.freeze_folder, '%s.pbtxt' % name, as_text=True)
    tf.io.write_graph(constant_graph, config.freeze_folder, '%s.pb' % name, as_text=False)
    print(colored("[INFO] convert model is success, saved at: ", "cyan", attrs=['bold']))
    print(config.freeze_folder + '/%s.pb' % name)


if __name__ == '__main__':
    main()
