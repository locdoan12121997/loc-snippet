from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
import keras
import sys
import cv2
import efficientnet.tfkeras as efficientnet

def get_model(height=112, width=112):
    base_model = efficientnet.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(height, width, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(8, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = sys.argv[1]
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = cv2.resize(im, (112, 112))
    img = np.expand_dims(im, 0)
    return img


def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    nb_classes = 8

    loss = K.one_hot([category_index], nb_classes) * model.layers[0].output)
    conv_output =  [l for l in model.layers[0].layers if l.name == layer_name][0].output


    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (112, 112))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

preprocessed_input = load_image(sys.argv[1])


# ['use_phone', 'drink', 'eat', 'cover_face', 'talk', 'sleep', 'record_pen', 'record_phone']

model = get_model()
model.load_weights('./2019_09_24_action_myData_effB1_fcl_affineScale_model.018.h5')

cam, heatmap = grad_cam(model, preprocessed_input, np.argmax(model.predict(preprocessed_input)), "top_activation")
cv2.imwrite("gradcam.jpg", cam)
