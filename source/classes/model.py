import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from source.enums import Models
from source.utils import prettify_datetime
from source.classes.models.vgg16 import VGG16
from source.classes.models.resnet import ResNet
from source.classes.dropout import DynamicDropout
from source.classes.l1l2 import DynamicL1L2
from source.classes.callbacks import DropoutParameterCallback


class CNNModel:
    def __init__(
        self,
        model_name: str,
        input_shape: list,
        n_classes: int,
        loss: str or object,
        optimizer: str or object,
        dropout: float,
        learning_rate: float,
        save_prefix: str,
        save_dir: str,
        is_restore: bool = False,
        is_dynamic: bool = False,
    ):
        self.model = None
        self.parameters = []
        self.history = None

        self.model_name = model_name
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.loss = loss

        self.save_prefix = save_prefix
        self.is_restore = is_restore
        self.is_dynamic = is_dynamic
        self.save_dir = save_dir

        self.callbacks = [
            # keras.callbacks.ModelCheckpoint(
            #     filepath=self.save_dir + '_{epoch:04d}-{val_loss:.2f}.hdf5',
            #     period=250,
            # ),
            # keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20),
        ]

        print("Chosen model: ", self.model_name)
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

                print('Restoring from', latest_checkpoint)
                self.model = keras.models.load_model(
                    latest_checkpoint,
                    custom_objects={
                        DynamicDropout.__name__: DynamicDropout,
                        DynamicL1L2.__name__: DynamicL1L2,
                    },
                )
        else:
            print('Building a new model')
            self.build_model()

    def build_model(self):
        dropout = DynamicDropout(
            self.dropout,
            is_dynamic=self.is_dynamic,
        )

        if self.model_name == Models.VGG16.value:
            self.model = VGG16.build(dropout, self.input_shape, self.n_classes)
        if self.model_name == Models.ResNet.value:
            self.model = ResNet.build(dropout, self.input_shape, self.n_classes)

        optimizer = self.optimizer if isinstance(self.optimizer, str) else self.optimizer(self.learning_rate)
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
            dropout_callback = DropoutParameterCallback(self.model)
            self.history = self.model.fit(
                ds_train,
                epochs=n_epochs,
                validation_data=validation_data,
                initial_epoch=initial_epoch,
                callbacks=self.callbacks + [dropout_callback],
                verbose=verbose,
            ).history
            self.parameters = dropout_callback.parameters[:]

            if save:
                self.model.save(
                    self.save_dir + self.save_prefix + f'_{n_epochs}_{prettify_datetime(datetime.now())}.hdf5',
                )
