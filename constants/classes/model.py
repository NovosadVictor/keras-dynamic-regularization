import os
import sys
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tensorflow import keras
from tensorflow.keras.layers import \
    Dense, \
    Conv2D, \
    MaxPooling2D, \
    Flatten
from tensorflow.keras import Sequential

from constants.classes.dropout import DynamicDropout
from constants.classes.callbacks import DropoutParameterCallback
from constants.classes.l1l2 import DynamicL1L2


class CNNModel:
    def __init__(self, *args, **kwargs):
        self.model = None
        self.parameters = []
        self.history = None
        self.save_dir = f'{kwargs.get("is_dynamic_dropout", False)}'

        self.callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.save_dir + '_{epoch:04d}-{val_loss:.2f}.hdf5',
                period=5,
            ),
            # keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20),
        ]

        self.build_or_restore_model(*args, **kwargs)

    def build_or_restore_model(self, *args, **kwargs):
        if kwargs.get('is_restore', False):
            checkpoints = [
                os.path.join(self.save_dir, name)
                for name in os.listdir(self.save_dir)
            ]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                print('Restoring from', latest_checkpoint)
                self.model = keras.models.load_model(latest_checkpoint)

        print('Building a new model')
        self.build_model(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        dropout = DynamicDropout(
            0.3,
            is_dynamic=kwargs.get('is_dynamic_dropout', False),
        )
        regularizer = DynamicL1L2()

        self.model = Sequential([
            Conv2D(input_shape=kwargs['input_shape'], filters=64, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            dropout,
            Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            dropout,
            Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            dropout,
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Flatten(),
            Dense(4096, activation='relu'),
            dropout,
            Dense(4096, activation='relu'),
            Dense(kwargs['n_classes'], activation='softmax')
        ])

        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'],
        )

    def fit(
        self,
        ds_train,
        n_epochs,
        validation_data=None,
        is_show=True,
    ):
        if self.model:
            dropout_callback = DropoutParameterCallback(self.model)
            self.history = self.model.fit(
                ds_train,
                epochs=n_epochs,
                validation_data=validation_data,
                callbacks=self.callbacks + [dropout_callback],
                verbose=0,
            ).history
            self.parameters = dropout_callback.parameters[:]

            if is_show:
              self.display()

    def display(self):
        if self.history:
            plt.figure(figsize=(12, 10))
            # summarize history for accuracy
            plt.plot(self.history['accuracy'])
            plt.plot(self.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train acc', 'test acc'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.figure(figsize=(12, 10))
            plt.plot(self.history['loss'])
            plt.plot(self.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train loss', 'test loss'], loc='upper left')
            plt.show()
