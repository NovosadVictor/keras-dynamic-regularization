import argparse
import json
from datetime import datetime
from matplotlib import pyplot as plt

import tensorflow as tf

from source.utils import load_tf_dataset, prettify_datetime
from source.constants import HISTORY_SAVE_FIR, MODEL_SAVE_FIR
from source.classes.model import CNNModel
from source.enums import Models, Databases, Losses, Optimizers


device_name = tf.test.gpu_device_name()


def train_model(
    database: str,
    input_shape: list,
    model_name: str,
    is_restore: bool = False,
    is_dynamic: bool = True,
    is_plot: bool = True,
    learning_rate: float = 0.001,
    dropout=0.3,
    loss=Losses.SparseCategoricalCrossentropy.value,
    optimizer=Optimizers.Adam,
):
    (ds_train, ds_test), ds_info = load_tf_dataset(database)
    n_classes = ds_info.features['label'].num_classes

    save_prefix = '{}_{}_{}_{}_{}_'.format(database, model_name, is_dynamic, dropout, learning_rate)

    with tf.device(device_name):
        model = CNNModel(
            n_classes=n_classes,
            is_restore=is_restore,
            model_name=model_name,
            input_shape=input_shape,
            is_dynamic=is_dynamic,
            save_dir=MODEL_SAVE_FIR,
            save_prefix=save_prefix,
            learning_rate=learning_rate,
            loss=loss,
            dropout=dropout,
            optimizer=optimizer,
        )
        model.fit(
            ds_train,
            n_epochs=30,
            validation_data=ds_test,
            initial_epoch=0,
            verbose=1,
        )

        score = model.model.evaluate(ds_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        history = model.history
        dropout_parameters = model.parameters

    now = prettify_datetime(datetime.now())
    with open(HISTORY_SAVE_FIR + '{}_history_{}.json'.format(save_prefix, now), 'w') as histories_file:
        json.dump(history, histories_file, indent=4)
    with open(HISTORY_SAVE_FIR + '{}_parameters_{}.json'.format(save_prefix, now), 'w') as parameters_file:
        json.dump(dropout_parameters, parameters_file, indent=4)

    if is_plot:
        plt.figure(figsize=(12, 10))

        # summarize history for accuracy
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--is_dynamic_dropout',
        type=int,
        help='if dynamic',
        default=0,
    )
    parser.add_argument(
        '--is_restore',
        type=int,
        help='if restore',
        default=0,
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help='vgg16 or resnet or ... (default - {})'.format(Models.VGG16.value),
        default=Models.VGG16.value,
    )
    parser.add_argument(
        '--database',
        type=str,
        help='mnist or cifar10 or ..., (default - {})'.format(Databases.Cifar10.value),
        default=Databases.Cifar10.value,
    )

    args = parser.parse_args()

    train_model(args.database, [32, 32, 3], args.model_name)
