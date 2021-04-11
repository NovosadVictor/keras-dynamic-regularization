import logging
import json
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import backend as K

from src.utils import prettify_datetime
from src.constants import HISTORY_SAVE_DIR
from .dynamic_dropouts import DynamicDropout, DynamicCrossmapDropBlock


class DropoutParameterCallback(keras.callbacks.Callback):
    def __init__(self, model, save_prefix, freq: int = 1, save_freq: int = 25, linear_scheduling: dict = None):
        self.model = model
        self.save_prefix = save_prefix
        self.freq = freq
        self.save_freq = save_freq
        self.linear_scheduling = linear_scheduling
        self.parameters = []
        super(DropoutParameterCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if self.linear_scheduling:
            rate = self.linear_scheduling['rate']
            init_value = self.linear_scheduling['init_value']
            if epoch > 0:
                # rate linear increasing
                # comment for dynamic dropout
                for index, layer in enumerate(self.model.layers):
                    if isinstance(layer, (DynamicDropout, DynamicCrossmapDropBlock)):
                        layer.set_rate(init_value + rate * epoch)

        if epoch % self.freq == 0:
            logging.info(
                'The average loss for epoch {} is {:7.2f}'.format(
                    epoch,
                    logs['loss'],
                ),
            )

            rates = []
            for index, layer in enumerate(self.model.layers):
                if isinstance(layer, (DynamicDropout, DynamicCrossmapDropBlock)):
                    rates.append(float(K.get_value(layer.rate)))

            logging.info('rates: %s', rates)
            self.parameters.append({'epoch': int(epoch), 'rates': rates})
        if epoch > 0 and epoch % self.save_freq == 0:
            now = prettify_datetime(datetime.now())
            with open(
                HISTORY_SAVE_DIR + '{}/{}-{}_parameters.json'.format(self.save_prefix, epoch, now),
                'w',
            ) as parameters_file:
                json.dump(self.parameters, parameters_file, indent=4)
