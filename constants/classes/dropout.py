from keras.layers import Dropout
from keras import backend as K


class DynamicDropout(Dropout):
    def __init__(self, rate, *args, is_dynamic=True, **kwargs):
        super(DynamicDropout, self).__init__(rate, *args, **kwargs)

        self.is_dynamic = is_dynamic
        if self.is_dynamic:
            self.rate = K.variable(self.rate, name='rate')

        self.rate_value = rate

    def set_dropout(self, rate):
        K.set_value(self.rate, rate)
        self.rate_value = rate

    def call(self, inputs, training=None):
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
        config = {}
        if self.is_dynamic:
            config['rate'] = K.get_value(self.rate)

        base_config = super(DynamicDropout, self).get_config()
        print(base_config)
        return {
            **base_config,
            **config,
        }
