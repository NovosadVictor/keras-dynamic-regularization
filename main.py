import os
import json
import logging
import pandas as pd
from pprint import pformat
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds

from src.helpers import make_sure_dir_exists
from src.utils import \
    load_tf_dataset, \
    prettify_datetime
from src.constants import \
    BASE_DIR, \
    DATABASES_DIR, \
    HISTORY_SAVE_DIR, \
    MODEL_SAVE_DIR, \
    MODELS, \
    DATABASES, \
    LOSSES, \
    OPTIMIZERS, \
    DROPOUT_CLASSES
from src.classes.model import CNNModel

logging.getLogger().setLevel(logging.INFO)


device_name = tf.test.gpu_device_name()


def train_model(**kwargs):
    save_prefix = kwargs['save_prefix']

    with open(f'{HISTORY_SAVE_DIR}/{save_prefix}/config.json', 'w') as f:
        json.dump(kwargs, f)
    with open(f'{MODEL_SAVE_DIR}/{save_prefix}/config.json', 'w') as f:
        json.dump(kwargs, f)

    model_config: dict = kwargs['model']
    database: dict = kwargs['database']
    learning_rate: dict = kwargs['learning_rate']
    dropout: dict = kwargs['dropout']

    model_name = MODELS[model_config.get('name', 'resnet')]
    is_restore = model_config.get('restore', False)
    restore_dir = model_config.get('restore_dir', False)
    loss = LOSSES[model_config.get('loss', 'categorical_crossentropy')]
    optimizer = OPTIMIZERS[model_config.get('optimizer', 'adam')]
    n_epochs = model_config.get('n_epochs', 200)

    input_shape = database.get('input_shape', [32, 32, 3])
    database_name = DATABASES[database.get('name', 'cifar10')]
    database_params = database.get('kwargs', {})

    learning_rate_init = learning_rate.get('init', 0.001)
    learning_rate_change_rate = learning_rate.get('change_rate', 0.01)

    dropout_rate = dropout.get('rate', 0.2)
    dropout_class = DROPOUT_CLASSES[dropout.get('class', 'crossmap_drop_block')]
    is_dynamic = dropout.get('is_dynamic', True)
    dropout_block_size = dropout.get('block_size', 3)

    if database_params:
        data_dir = database_params['data_dir']

        write_dir = make_sure_dir_exists(f'{DATABASES_DIR}/{database_name}')
        make_sure_dir_exists(f'{write_dir}/extracted')
        make_sure_dir_exists(f'{write_dir}/downloaded')
        make_sure_dir_exists(f'{write_dir}/data')

        download_config = tfds.download.DownloadConfig(
            extract_dir=f'{write_dir}/extracted',
            manual_dir=data_dir,
        )
        download_and_prepare_kwargs = {
            'download_dir': f'{write_dir}/downloaded',
            'download_config': download_config,
        }
        database_params = {
            'data_dir': f'{write_dir}/data',
            'download_and_prepare_kwargs': download_and_prepare_kwargs,
        }

    (ds_train, ds_test), ds_info = load_tf_dataset(database_name, **database_params)
    n_classes = ds_info.features['label'].num_classes

    if kwargs.get('multiple_gpu', False):
        scope = tf.distribute.MirroredStrategy().scope
    else:
        scope = lambda: tf.device(tf.test.gpu_device_name())

    with scope():
        model = CNNModel(
            model={
                'n_classes': n_classes,
                'is_restore': is_restore,
                'restore_dir': restore_dir,
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
            verbose=1,
        )

    score = model.model.evaluate(ds_test, verbose=0)
    logging.info('Test loss: %s', score[0])
    logging.info('Test accuracy: %s', score[1])

    history = model.history
    dropout_parameters = model.parameters

    now = prettify_datetime(datetime.now())
    pd.DataFrame(history).to_csv(HISTORY_SAVE_DIR + '{}/history_{}.csv'.format(save_prefix, now))
    pd.DataFrame(dropout_parameters).to_csv(HISTORY_SAVE_DIR + '{}/parameters_{}.csv'.format(save_prefix, now))


if __name__ == '__main__':
    now = prettify_datetime(datetime.now())
    logging.info(now)
    os.makedirs(f'{HISTORY_SAVE_DIR}/{now}')
    os.makedirs(f'{MODEL_SAVE_DIR}/{now}')

    with open(f'{BASE_DIR}/config.json', 'r') as config_file:
        config = json.load(config_file)

    logging.info('Chosen config:')
    logging.info(pformat(config, indent=4))

    train_model(
        **config,
        save_prefix=f'{now}/'
    )
