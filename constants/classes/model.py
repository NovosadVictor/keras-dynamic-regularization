import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from keras.layers import Dense
from keras import Sequential
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects

from constants.utils import loss_wrapper
from constants.classes.dropout import DynamicDropout
from constants.classes.l1l2 import DynamicL1L2
get_custom_objects().update({DynamicL1L2.__name__: DynamicL1L2})
get_custom_objects().update({DynamicDropout.__name__: DynamicDropout})


class NNModel:
    def __init__(self):
        self.model = None
        self.build_model()

    def build_model(self):
        c, p0 = K.variable(0.1), 0.1
        l1, l2 = 0, K.get_value(c)
        dropout = DynamicDropout(0.01)
        regularizer = DynamicL1L2(l1, l2)

        self.model = Sequential()
        self.model.add(Dense(
            16,
            input_dim=8,
            activation='relu',
        ))
        self.model.add(Dense(
            32,
            activation='sigmoid',
        ))
        self.model.add(Dense(
            8,
            activation='relu',
        ))
        self.model.add(Dense(
            1,
            activation='sigmoid',
        ))
        self.model.compile(
            loss=loss_wrapper(dropout),
            optimizer='adam',
            metrics=['accuracy'],
        )

    def fit(self, x_train, y_train, n_epochs, validation_split=0, batch_size=128):
        self.model.fit(
            x_train,
            y_train,
            epochs=n_epochs,
            validation_split=validation_split,
            batch_size=batch_size,
        )
