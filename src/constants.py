from tensorflow.keras.optimizers import Adam

from .classes.dynamic_dropouts import DynamicDropout, DynamicCrossmapDropBlock

from .helpers import make_sure_dir_exists


BASE_DIR = '.'

DATABASES_DIR = make_sure_dir_exists(f'{BASE_DIR}/databases/')
MODEL_SAVE_DIR = make_sure_dir_exists(f'{BASE_DIR}/models/')
HISTORY_SAVE_DIR = make_sure_dir_exists(f'{BASE_DIR}/histories/')


MODELS = {
    'resnet': 'res_net',
}


DATABASES = {
    'cifar10': 'cifar10',
    'imagenet': 'imagenet',
}


LOSSES = {
    'categorical_crossentropy': 'categorical_crossentropy',
}


OPTIMIZERS = {
    'adam': Adam,
}


DROPOUT_CLASSES = {
    'standard': DynamicDropout,
    'crossmap_drop_block': DynamicCrossmapDropBlock,
}
