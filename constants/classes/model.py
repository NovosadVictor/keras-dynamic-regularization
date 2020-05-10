import os
import sys
from datetime import datetime
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import keras
from keras.layers import \
    Dense, \
    Conv2D, \
    MaxPooling2D, \
    Flatten
from keras import Sequential
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects

from constants.utils import loss_wrapper, prettify_datetime
from constants.classes.dropout import DynamicDropout
from constants.classes.callbacks import DropoutParameterCallback
from constants.classes.l1l2 import DynamicL1L2
get_custom_objects().update({DynamicL1L2.__name__: DynamicL1L2})
get_custom_objects().update({DynamicDropout.__name__: DynamicDropout})


class CNNModel:
    def __init__(self, *args, **kwargs):
        self.model = None
        self.history = None
        self.checkpoint_dir = 'checkpoints/'
        self.plots_dir = 'plots/'

        self.callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_dir + '{epoch:02d}-{val_loss:.2f}.hdf5',
                period=1,
            ),
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20),
        ]

        self.build_or_restore_model(*args, **kwargs)

    def build_or_restore_model(self, *args, **kwargs):
        if kwargs.get('is_restore', False):
            checkpoints = [
                os.path.join(self.checkpoint_dir, name)
                for name in os.listdir(self.checkpoint_dir)
            ]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                print('Restoring from', latest_checkpoint)
                self.model = keras.models.load_model(latest_checkpoint)

        print('Building a new model')
        self.build_model(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        dropout = DynamicDropout(
            0.5,
            is_dynamic=kwargs.get('is_dynamic_dropout', False),
        )
        regularizer = DynamicL1L2()

        self.model = Sequential([
            Conv2D(32, (3, 3), input_shape=kwargs['input_shape'], padding='same', activation='relu'),
            dropout,
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(),
            dropout,
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            dropout,
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(),
            dropout,
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            dropout,
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(),
            dropout,
            Flatten(),
            Dense(1024, activation='relu'),
            dropout,
            Dense(512, activation='relu'),
            Dense(kwargs['n_classes'], activation='softmax')
        ])

        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'],
        )

    def fit(
            self,
            x_train, y_train,
            n_epochs, validation_split=0,
            validation_data=None, batch_size=128,
            is_show=True,
    ):
        if self.model:
            self.history = self.model.fit(
                x_train,
                y_train,
                epochs=n_epochs,
                validation_split=validation_split,
                validation_data=validation_data,
                batch_size=batch_size,
                callbacks=self.callbacks + [DropoutParameterCallback(self.model)],
                verbose=0,
            ).history

            if is_show:
                self.display()

    def display(self):
        if self.history:
            now = prettify_datetime(datetime.now())
            os.mkdir(os.path.join('plots', now))

            plt.figure(figsize=(12, 10))
            # summarize history for accuracy
            plt.plot(self.history['accuracy'])
            plt.plot(self.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train acc', 'test acc'], loc='upper left')
            plt.savefig(f'{self.plots_dir}/{now}/accuracy.png')
            # summarize history for loss
            plt.figure(figsize=(12, 10))
            plt.plot(self.history['loss'])
            plt.plot(self.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train loss', 'test loss'], loc='upper left')
            plt.savefig(f'{self.plots_dir}/{now}/accuracy.png')
