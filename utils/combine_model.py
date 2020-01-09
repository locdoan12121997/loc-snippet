"""
aioz.aiar.truongle - Sep 25, 2019
combine multi model the same input
"""
import os,sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
from config import Config
from termcolor import colored
from tensorflow.python.framework import graph_util
from tensorflow.python.keras import models, layers, backend


class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


class CombineWrapper:
    def __init__(self, graph="graph/combine.pb", memory_fraction=0.7):
        self.graph_fb = graph
        # # Config for GPU
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        self.config.gpu_options.allow_growth = True
        self.config.log_device_placement = False
        self.sess = None

        self.__load_graph()
        self.__init_prediction()

    def __load_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_fb, 'rb') as f:
                graph_def.ParseFromString(f.read())
            with self.graph.as_default():
                tf.import_graph_def(graph_def)
        # tf.get_default_graph().finalize()

    def __init_prediction(self):
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph, config=self.config)
            self.input_tensor = self.graph.get_tensor_by_name('import/input_1_3:0')
            self.output_tensor = self.graph.get_tensor_by_name('import/concatenate/concat:0')
        # print(colored("[INFO] Init model is done", "green", attrs=['bold']))

    def predict(self, images):
        outputs = self.sess.run([self.output_tensor], feed_dict={self.input_tensor: images})
        return np.asarray(outputs)[0]


def load_model(model_pth, name):
    def swish(x):
        return tf.nn.swish(x)
    model = models.load_model(model_pth,
                              custom_objects={'swish': swish, 'FixedDropout': FixedDropout},
                              compile=False)
    for layer in model.layers:
        layer.trainable = False

    # RENAME
    model._name = name
    return model


def main():
    config = Config()
    # LOAD MODEL
    backend.set_learning_phase(0)
    age_model = load_model(model_pth=config.age_h5_path, name="age_model")
    gender_model = load_model(model_pth=config.gender_h5_path, name="gender_model")
    emotion_model = load_model(model_pth=config.expr_h5_path, name="emotion_model")

    # COMBINE
    _, height, width, depth = age_model.input.shape
    cb_input = layers.Input(shape=(height, width, depth))
    age_outs = age_model(cb_input)
    gender_outs = gender_model(cb_input)
    emo_outs = emotion_model(cb_input)
    merged = layers.Concatenate()([age_outs, gender_outs, emo_outs])
    cb_model = models.Model(inputs=cb_input, outputs=merged)
    cb_model.summary()

    # SAVE MODEL
    cb_model.save(config.combine_model_h5_path)
    print(colored("[INFO] Combine model is DONE, saved at </ {} /> ".format(config.combine_model_h5_path),
                  color='red', attrs=['bold']))

    # FREEZE
    sess = backend.get_session()
    converted_output_node_names = [node.op.name for node in cb_model.outputs]
    print("[INFO] in: ", cb_model.inputs)  # input_1_4:0
    print("[INFO] out: ", cb_model.outputs)  # concatenate/concat:0
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        converted_output_node_names)

    freeze_folder = config.freeze_folder
    name = os.path.split(config.combine_model_pb_path)[-1].split('.')[0]
    tf.train.write_graph(constant_graph, freeze_folder, '%s.pbtxt' % name, as_text=True)
    tf.train.write_graph(constant_graph, freeze_folder, '%s.pb' % name, as_text=False)
    print(colored("[INFO] convert model is success, saved at </ %s/%s.pb />" % (freeze_folder, name),
                  color="cyan", attrs=['bold']))


def test_model(use_h5=True):
    config = Config()
    ims = np.ones((7, 112, 112, 3))
    if use_h5:
        """test with h5"""
        def swish(x):
            return tf.nn.swish(x)

        model = models.load_model(config.combine_model_h5_path,
                                  custom_objects={'swish': swish})

        outs = model.predict(ims)
    else:
        """test with pb"""
        graph = config.combine_model_pb_path
        cb = CombineWrapper(graph=graph)
        outs = cb.predict(ims)
    print("[INFO] Outputs: ", outs)


if __name__ == '__main__':
    main()
    # test_model(use_h5=False)

