from tensorflow import keras
from tensorflow.keras import backend as K

from .dropout import DynamicDropout


class DropoutParameterCallback(keras.callbacks.Callback):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.parameters = []
        super(DropoutParameterCallback, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            print(
                'The average loss for epoch {} is {:7.2f}'.format(
                  epoch,
                  logs['loss'],
                ),
            )
            for index, layer in enumerate(self.model.layers):
                if isinstance(layer, DynamicDropout):
                    rate = layer.rate
                    rate = K.get_value(rate)
                    self.parameters.append(rate)
                    break
