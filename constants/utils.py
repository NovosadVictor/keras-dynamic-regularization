from datetime import datetime
from numpy import loadtxt

import keras
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.datasets import mnist

from .classes.dropout import DynamicDropout
from .classes.l1l2 import DynamicL1L2


def prettify_datetime(time: datetime) -> str:
    return time.strftime('%m.%d.%Y_%H:%M:%S')


def set_model_l1_l2(model, l1=0, l2=0.01):
    for layer in model.layers:
        if 'kernel_regularizer' in dir(layer) and isinstance(layer.kernel_regularizer, DynamicL1L2):
            layer.kernel_regularizer.set_l1_l2(l1, l2)


def set_model_dropout(model, dropout=0.5):
    for layer in model.layers:
        if isinstance(layer, DynamicDropout):
            layer.set_dropout(dropout)


def load_diabetes_dataset():
    dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
    X = dataset[:, 0: 8]
    y = dataset[:, 8]

    return X, y


def load_mnist(img_rows, img_cols, n_classes):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    return input_shape, x_train, y_train, x_test, y_test


def loss_wrapper(dropout_layer):
    def loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred)

    return loss
