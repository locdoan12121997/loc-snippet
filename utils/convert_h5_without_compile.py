"""
aioz.aiar.truongle - Sep 25, 2019
convert h5 without compile
"""
import tensorflow as tf
from tensorflow.python.keras import models
from config import Config


def main():
    config = Config()

    # LOAD MODEL
    def swish(x):
        return tf.nn.swish(x)

    model_fn = models.load_model(config.age_h5_checkpoint,
                                 custom_objects={'swish': swish, 'tf': tf},
                                 compile=False)
    # model_fn.summary()
    # SET TRAINABLE IS FALSE
    for layer in model_fn.layers:
        layer.trainable = False

    model_fn.save(config.age_h5_checkpoint_convert)


if __name__ == '__main__':
    main()
