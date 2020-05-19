from datetime import datetime
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.losses import binary_crossentropy

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


def loss_wrapper(dropout_layer):
    def loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred)

    return loss


def resize_img(image, label, size):
    return tf.image.resize(image, size), label


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def categorize_label(image, label, n_classes):
    label = tf.one_hot(tf.cast(label, tf.int32), n_classes)
    label = tf.cast(label, tf.float32)
    return image, label


def load_tf_dataset(tf_dataset_name: str, img_size: list = None):
    (ds_train, ds_test), ds_info = tfds.load(
        tf_dataset_name,
        split=['train', 'test'],
        with_info=True,
        shuffle_files=True,
        as_supervised=True,
    )
    n_classes = ds_info.features['label'].num_classes
    categorize = partial(categorize_label, n_classes=n_classes)

    # train
    ds_train = ds_train.map(
        normalize_img,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    ds_train = ds_train.map(
        categorize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # test
    ds_test = ds_test.map(
        normalize_img,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds_test = ds_test.map(
        categorize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    if img_size:
        resize_image = partial(resize_img, size=img_size[:2])
        ds_train = ds_train.map(
            resize_image,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        ds_test = ds_test.map(
            resize_image,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    return (ds_train, ds_test), ds_info
