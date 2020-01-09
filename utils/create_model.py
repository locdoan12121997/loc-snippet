"""
aioz.aiar.truongle - Sep 26, 2019
get model keras for load weight
"""
import tensorflow as tf
from tensorflow.python.keras import models, layers
# from tensorflow.python.keras.applications import efficientnet
from src.gender_estimation.helpers.efficientnet import efficientnet
from tensorflow.python.keras import backend as K
from termcolor import colored
from tensorflow.python.framework import graph_util


class EmotionModel:
    def __init__(self):
        self.input_shape = (112, 112, 3)
        self.num_class = 6
        self.model_fn = None
        self._create_model()

    def _create_model(self):
        base_model = efficientnet.EfficientNetB1(weights='imagenet', include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        predictions = layers.Dense(self.num_class, activation='softmax')(x)
        self.model_fn = models.Model(inputs=base_model.input, outputs=predictions, name="emotion_effB1")

    def get_model(self):
        return self.model_fn


class AgeModel:
    def __init__(self):
        self.input_shape = (112, 112, 3)
        self.range_age = [15, 69]
        self.model_fn = None
        self._create_model()

    def _create_model(self):
        num_dense = self.range_age[1] - self.range_age[0] + 1  # 15 -> 69 is 55 class
        base_model = efficientnet.EfficientNetB1(weights='imagenet', include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        prob_classes = layers.Dense(num_dense, activation='softmax')(x)
        # Calculate mean output
        # # Why we need Lambda Layer here
        start = int(self.range_age[0])  # can't load model if put args into layers.Lambda
        stop = int(self.range_age[1] + 1)
        a_range = layers.Lambda(lambda z: K.arange(start=start, stop=stop, step=1, dtype='float32'))(prob_classes)
        # Lambda_2 = Lambda_1 * dense [output from 15 to 69]
        mul = layers.Lambda(lambda a: tf.multiply(a[0], a[1]))([prob_classes, a_range])
        # Lambda3 = prob_s*ages
        pre_a = layers.Lambda(lambda z: tf.reduce_sum(z, -1, keepdims=True))(mul)
        self.model_fn = models.Model(inputs=base_model.input, outputs=pre_a, name="age_effB1")

    def get_model(self):
        return self.model_fn


class GenderModel:
    def __init__(self):
        self.input_shape = (112, 112, 3)
        self.model_fn = None
        self._create_model()

    def _create_model(self):
        base_model = efficientnet.EfficientNetB4(weights='imagenet', include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        predictions = tf.keras.layers.Dense(1)(x)

        def gumbel_sigmoid(x):
            U1 = tf.random_uniform(tf.shape(x), 0, 1)
            U2 = tf.random_uniform(tf.shape(x), 0, 1)
            noise = -tf.log(tf.log(U2 + 1e-20) / tf.log(U1 + 1e-20) + 1e-20)
            noise = noise * tf.cast(tf.keras.backend.learning_phase(), tf.float32)
            T = 0.8 ** (tf.cast(tf.keras.backend.learning_phase(), tf.float32))
            return tf.keras.backend.sigmoid((x + noise) / T)

        predictions = tf.keras.layers.Lambda(gumbel_sigmoid)(predictions)

        self.model_fn = models.Model(inputs=base_model.input, outputs=predictions, name="Gender_effB4")

    def get_model(self):
        return self.model_fn


# TEST
if __name__ == '__main__':
    # FREEZE
    K.set_learning_phase(0)
    # EMOTION
    emotion_model = EmotionModel()
    emotion_model = emotion_model.get_model()
    emo_w = "inference_models/models_pb/weights_only/2019_09_24_FER_effb1_fcl_affineScale_model.h5"
    emotion_model.load_weights(emo_w)
    # emo_model.summary()

    # AGE
    age_model = AgeModel()
    age_model = age_model.get_model()
    age_w = "inference_models/models_pb/weights_only/2019_09_23_age_EfficientNetB1_MegaAgeAsian_Multisacle_HeavyAug.h5"
    age_model.load_weights(age_w)
    # age_model.summary()

    # GENDER
    gender_model = GenderModel()
    gender_model = gender_model.get_model()
    gen_w = "inference_models/models_pb/weights_only/2019_09_13_gender_EfficientNetB4_model.h5"
    gender_model.load_weights(gen_w)
    # age_model.summary()

    # COMBINE
    _, height, width, depth = age_model.input.shape
    cb_input = layers.Input(shape=(height, width, depth))
    age_outs = age_model(cb_input)
    gen_outs = gender_model(cb_input)
    emo_outs = emotion_model(cb_input)
    merged = layers.Concatenate()([age_outs, gen_outs, emo_outs])
    cb_model = models.Model(inputs=cb_input, outputs=merged, name="age_gen_emo_model")
    cb_model.summary()

    sess = K.get_session()
    converted_output_node_names = [node.op.name for node in cb_model.outputs]
    print("[INFO] in: ", cb_model.inputs)  # input_1_4:0
    print("[INFO] out: ", cb_model.outputs)  # concatenate/concat:0
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        converted_output_node_names)

    freeze_folder = "inference_models/models_pb/weights_only"
    name = "2019_09_26_age_gen_emo"
    tf.train.write_graph(constant_graph, freeze_folder, '%s.pbtxt' % name, as_text=True)
    tf.train.write_graph(constant_graph, freeze_folder, '%s.pb' % name, as_text=False)
    print(colored("[INFO] convert model is success, saved at </ %s/%s.pb />" % (freeze_folder, name),
    color="cyan", attrs=['bold']))
