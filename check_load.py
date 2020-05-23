import sys
import json
import logging
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group, load_optimizer_weights_from_hdf5_group
from tensorflow.python.keras.saving import saving_utils
import h5py

from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K


class DynamicDropout(Dropout):
    def __init__(self, rate=0.5, *args, is_dynamic=True, **kwargs):
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


def load_model():
    filepath = sys.argv[1]
    f = h5py.File(filepath, mode='r')
    print(1)
    model_config = f.attrs.get('model_config')
    print(2)
    model_config = json.loads(model_config.decode('utf-8'))
    print(3)
    custom_objects = {'DynamicDropout': DynamicDropout}
    with generic_utils.CustomObjectScope(custom_objects or {}):
        model = model_config_lib.model_from_config(
            model_config,
            custom_objects=custom_objects,
        )
        print(4)
        load_weights_from_hdf5_group(f['model_weights'], model.layers)
        print(5)

        if compile:
            # instantiate optimizer
            training_config = f.attrs.get('training_config')
            if training_config is None:
                logging.warning('No training configuration found in the save file, so '
                                'the model was *not* compiled. Compile it manually.')
                return model
            training_config = json.loads(training_config.decode('utf-8'))

            # Compile model.
            model.compile(**saving_utils.compile_args_from_training_config(
                training_config, custom_objects))

            # Set optimizer weights.
            if 'optimizer_weights' in f:
                try:
                    model.optimizer._create_all_weights(model.trainable_variables)
                except (NotImplementedError, AttributeError):
                    logging.warning(
                        'Error when creating the weights of optimizer {}, making it '
                        'impossible to restore the saved optimizer state. As a result, '
                        'your model is starting with a freshly initialized optimizer.')

                optimizer_weight_values = load_optimizer_weights_from_hdf5_group(f)
                try:
                    model.optimizer.set_weights(optimizer_weight_values)
                except ValueError:
                    logging.warning('Error in loading the saved optimizer '
                                    'state. As a result, your model is '
                                    'starting with a freshly initialized '
                                    'optimizer.')


load_model()
print(6)
