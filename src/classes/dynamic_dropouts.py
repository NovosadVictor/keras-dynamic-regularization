from uuid import uuid4

import tensorflow as tf
from tensorflow.keras.layers import \
    Layer, \
    Dropout
from tensorflow.keras import backend as K


class DynamicDropout(Dropout):
    def __init__(self, rate=0.5, is_dynamic=True, **kwargs):
        super(DynamicDropout, self).__init__(rate)

        if is_dynamic:
            self.rate = K.variable(self.rate, name='rate_' + str(uuid4()))
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


class CrossmapDropBlock(Layer):
    def __init__(self, rate, block_size, scale=True):
        super(CrossmapDropBlock, self).__init__()
        self.rate = rate

        self.block_size = int(block_size)
        self.scale = scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channels = input_shape.as_list()

        # for small shapes
        if self.block_size >= self.h // 4 or self.block_size >= self.w // 4:
            self.block_size = 1

        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]

        super(CrossmapDropBlock, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if 0. < self.rate_value < 1.:
            def drop():
                sampling_mask_shape = tf.stack([tf.shape(inputs)[0],
                                                self.h - self.block_size + 1,
                                                self.w - self.block_size + 1,
                                                ])
                # self.channels])
                gamma = self.rate * (self.w * self.h) / (self.block_size ** 2) / \
                        ((self.w - self.block_size + 1) * (self.h - self.block_size + 1))
                self.gamma = gamma
                mask = tf.cast(gamma >= tf.random.uniform(sampling_mask_shape, minval=0, maxval=1, dtype=tf.float32),
                               dtype=tf.float32)
                mask = tf.stack([mask for _ in range(self.channels)], axis=-1)
                mask = tf.pad(mask, self.padding)
                mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
                mask = 1 - mask

                output = inputs * tf.cast(mask, dtype=tf.float32) / (1 - self.rate) # * tf.cast(tf.size(mask), dtype=tf.float32) / tf.reduce_sum(mask)

                return output

            if training is None:
                training = K.learning_phase()

            output = tf.cond(tf.logical_not(tf.cast(training, dtype=tf.bool)),
                             true_fn=lambda: inputs,
                             false_fn=drop)
            return output

        return inputs

    def get_config(self):
        config = super(CrossmapDropBlock, self).get_config()
        config = {
            **config,
            **{
                'rate': self.rate,
                'block_size': self.block_size,
                'scale': self.scale,
            },
        }

        return config


class DynamicCrossmapDropBlock(CrossmapDropBlock):
    def __init__(self, rate, block_size, is_dynamic=True, **kwargs):
        super(DynamicCrossmapDropBlock, self).__init__(rate, block_size)

        self.rate_value = self.rate
        if is_dynamic:
            self.rate = K.variable(self.rate, name='rate_' + str(uuid4()))

        self.is_dynamic = is_dynamic

    def set_rate(self, rate=None):
        if rate is not None and 0. < rate < 1.:
            if self.is_dynamic:
                K.set_value(self.rate, rate)
            else:
                self.rate = rate

            self.rate_value = rate

    def get_config(self):
        config = super(DynamicCrossmapDropBlock, self).get_config()
        if self.is_dynamic:
            config['rate'] = K.get_value(self.rate)

        return config
