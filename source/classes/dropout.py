from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K


class DynamicDropout(Dropout):
    def __init__(self, rate=0.5, *args, is_dynamic=True, **kwargs):
        super(DynamicDropout, self).__init__(rate, *args, **kwargs)

        if is_dynamic:
            self.rate = K.variable(self.rate, name='rate')
            self.rate_value = rate

        self.is_dynamic = is_dynamic

    def set_dropout(self, rate):
        K.set_value(self.rate, rate)
        self.rate_value = rate

    def call(self, inputs, training=None):
        if not self.is_dynamic:
            return super(DynamicDropout, self).call(inputs, training)

        if 0. < self.rate_value < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(
                    inputs,
                    self.rate,
                    noise_shape,
                    seed=self.seed,
                )

            return K.in_train_phase(
                dropped_inputs,
                inputs,
                training=training,
            )
        return inputs

    def get_config(self):
        config = super(DynamicDropout, self).get_config()
        if self.is_dynamic:
            config['rate'] = K.get_value(self.rate)

        return config
