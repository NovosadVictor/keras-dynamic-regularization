import os
import logging
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from src.utils import prettify_datetime, learning_rate_scheduler
from src.constants import HISTORY_SAVE_DIR
from src.classes.models.resnet import ResNet
from src.classes.dynamic_dropouts import DynamicDropout, DynamicCrossmapDropBlock
from src.classes.callbacks import DropoutParameterCallback


class CNNModel:
    def __init__(
            self,
            model: dict,
            dropout: dict,
            learning_rate: dict,
            loss: str or object,
            optimizer: str or object,
            save_prefix: str,
            save_dir: str,

    ):
        self.model = None
        self.parameters = []
        self.history = None

        self.learning_rate = learning_rate
        self.dropout = dropout

        self.model_name = model['model_name']
        self.input_shape = model['input_shape']
        self.n_classes = model['n_classes']

        self.optimizer = optimizer
        self.loss = loss

        self.save_prefix = save_prefix
        self.is_restore = model['is_restore']
        self.save_dir = save_dir


        self.callbacks = [
            keras.callbacks.LearningRateScheduler(
                learning_rate_scheduler(init_rate=learning_rate['init'], k=learning_rate['change_rate']),
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=self.save_dir + self.save_prefix + '_{epoch:04d}-{val_loss:.2f}.hdf5',
                period=50,
            ),
        ]
        logging.info(self.is_restore)

        logging.info("Chosen model: %s", self.model_name)
        self.build_or_restore_model()

    def build_or_restore_model(self):
        if self.is_restore:
            checkpoints = [
                os.path.join(self.save_dir, name)
                for name in os.listdir(self.save_dir)
                if self.save_prefix in name
            ]

            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)

                logging.info('Restoring from %s', latest_checkpoint)
                self.model = keras.models.load_model(
                    latest_checkpoint,
                    custom_objects={
                        DynamicCrossmapDropBlock.__name__: DynamicCrossmapDropBlock,
                        DynamicDropout.__name__: DynamicDropout,
                    },
                )
        else:
            logging.info('Building a new model')
            self.build_model()

    def build_model(self):
        if self.model_name == 'resnet':
            self.model = ResNet.build(self.dropout, self.input_shape, self.n_classes)

        optimizer = self.optimizer if isinstance(self.optimizer, str) else self.optimizer(self.learning_rate['init'])
        self.model.compile(
            loss=self.loss,
            optimizer=optimizer,
            metrics=['accuracy'],
        )

    def fit(
            self,
            ds_train,
            n_epochs,
            validation_data=None,
            initial_epoch=0,
            verbose=1,
            save=True,
    ):
        if self.model:
            dropout_callback = DropoutParameterCallback(self.model, self.save_prefix)
            filename = HISTORY_SAVE_DIR + '{}_history.csv'.format(self.save_prefix)
            history_logger = tf.keras.callbacks.CSVLogger(filename, separator=',', append=True)

            self.history = self.model.fit(
                ds_train,
                epochs=n_epochs,
                validation_data=validation_data,
                initial_epoch=initial_epoch,
                callbacks=self.callbacks + [dropout_callback, history_logger],
                verbose=verbose,
            ).history
            self.parameters = dropout_callback.parameters[:]

            if save:
                self.model.save(
                    self.save_dir + self.save_prefix + f'_{n_epochs}_{prettify_datetime(datetime.now())}.hdf5',
                )
