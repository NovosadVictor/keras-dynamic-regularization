from enum import Enum

from tensorflow.keras.optimizers import Adam


class Models(Enum):
    VGG16 = 'vgg16'
    ResNet = 'res_net'


class Databases(Enum):
    Cifar10 = 'cifar10'


class Losses(Enum):
    CategoricalCrossentropy = 'categorical_crossentropy'
    SparseCategoricalCrossentropy = 'sparse_categorical_crossentropy'


class Optimizers(Enum):
    Adam = Adam
