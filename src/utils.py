from math import exp
from datetime import datetime
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds


def make_save_prefix_name(kwargs: dict) -> str:
    return '_'.join([f'{key}={str(value)}' for key, value in kwargs.items()])


def prettify_datetime(time: datetime) -> str:
    return time.strftime('%m.%d.%Y_%H:%M:%S')


def resize_img(image, label, size):
    return tf.image.resize(image, size), label


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def categorize_label(image, label, n_classes):
    label = tf.one_hot(tf.cast(label, tf.int32), n_classes)
    label = tf.cast(label, tf.float32)
    return image, label


def load_tf_dataset(
    tf_dataset_name: str,
    img_size: list = None,
    batch_size: int = 128,
    data_dir: str = None,
    download_and_prepare_kwargs: dict = None,
):
    (ds_train, ds_test), ds_info = tfds.load(
        tf_dataset_name,
        data_dir=data_dir,
        split=['train', 'test'],
        with_info=True,
        shuffle_files=True,
        as_supervised=True,
        download_and_prepare_kwargs=download_and_prepare_kwargs,
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
    ds_train = ds_train.batch(batch_size)
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
    ds_test = ds_test.batch(batch_size)
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


def learning_rate_scheduler(init_rate=0.001, k=0.01):
    def learning_rate_schedule(epoch, lr):
        return init_rate * exp(-k * epoch)

    return learning_rate_schedule
