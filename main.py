import json
import logging
from pprint import pformat
from datetime import datetime

import tensorflow as tf

from src.utils import \
    load_tf_dataset, \
    prettify_datetime, \
    make_save_prefix_name
from src.constants import \
    HISTORY_SAVE_DIR, \
    MODEL_SAVE_DIR, \
    MODELS, \
    DATABASES, \
    LOSSES, \
    OPTIMIZERS, \
    DROPOUT_CLASSES
from src.classes.model import CNNModel


device_name = tf.test.gpu_device_name()


def train_model(**kwargs):
    model = kwargs['model']
    database = kwargs['database']
    learning_rate = kwargs['learning_rate']
    dropout = kwargs['dropout']

    model_name = MODELS[model.get('name', 'resnet')]
    is_restore = model.get('restore', False)
    loss = LOSSES[model.get('loss', 'categorical_crossentropy')]
    optimizer = OPTIMIZERS[model.get('optimizer', 'adam')]
    n_epochs = model.get('n_epochs', 200)

    input_shape = database.get('input_shape', [32, 32, 3])
    database_name = DATABASES[database.get('name', 'cifar10')]
    database_params = database.get('kwargs', {})

    learning_rate_init = learning_rate.get('init', 0.001)
    learning_rate_change_rate = learning_rate.get('change_rate', 0.01)

    dropout_rate = dropout.get('rate', 0.2)
    dropout_class = DROPOUT_CLASSES[dropout_rate.get('class', 'crossmap_drop_block')]
    is_dynamic = dropout.get('is_dynamic', True)
    dropout_block_size = dropout.get('block_size', 3)

    (ds_train, ds_test), ds_info = load_tf_dataset(database_name, **(database_params or {}))
    n_classes = ds_info.features['label'].num_classes

    save_prefix = make_save_prefix_name(kwargs)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = CNNModel(
            model={
                'n_classes': n_classes,
                'is_restore': is_restore,
                'model_name': model_name,
                'input_shape': input_shape,
            },
            dropout={
                'class': dropout_class,
                'kwargs': {
                    'rate': dropout_rate,
                    'block_size': dropout_block_size,
                    'is_dynamic': is_dynamic,
                },
            },
            learning_rate={
                'init': learning_rate_init,
                'change_rate': learning_rate_change_rate,
            },
            save_dir=MODEL_SAVE_DIR,
            save_prefix=save_prefix,
            loss=loss,
            optimizer=optimizer,
        )

        model.fit(
            ds_train,
            n_epochs=n_epochs,
            validation_data=ds_test,
            initial_epoch=0,
            verbose=1,
        )

    score = model.model.evaluate(ds_test, verbose=0)
    logging.info('Test loss: %s', score[0])
    logging.info('Test accuracy: %s', score[1])

    history = model.history
    dropout_parameters = model.parameters

    now = prettify_datetime(datetime.now())
    with open(HISTORY_SAVE_DIR + '{}_history_{}.json'.format(save_prefix, now), 'w') as histories_file:
        json.dump(history, histories_file, indent=4)
    with open(HISTORY_SAVE_DIR + '{}_parameters_{}.json'.format(save_prefix, now), 'w') as parameters_file:
        json.dump(dropout_parameters, parameters_file, indent=4)


if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    logging.info('Chosen config:')
    logging.info(pformat(config))

    train_model(**config)
