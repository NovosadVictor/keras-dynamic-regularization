import keras
from keras import backend as K

from .dropout import DynamicDropout


class DropoutParameterCallback(keras.callbacks.Callback):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super(DropoutParameterCallback, self).__init__(*args, **kwargs)

    def on_train_batch_begin(self, batch, logs=None):
        for index, layer in enumerate(self.model.layers):
            if isinstance(layer, DynamicDropout):
                rate = layer.rate
                try:
                    # check if it is a tf.Variable
                    rate = K.get_value(rate)
                    print(f'Layer: {index}, rate {rate}')
                except Exception:
                    pass
