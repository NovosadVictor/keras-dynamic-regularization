from tensorflow.keras.layers import \
    Dense, \
    Conv2D, \
    MaxPooling2D, \
    Flatten
from tensorflow.keras import Sequential


class VGG16:
    @staticmethod
    def build(dropout, input_shape: list, n_classes: int) -> Sequential:
        return Sequential([
            Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            dropout,
            Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            dropout,
            Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            dropout,
            Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            dropout,
            Dense(4096, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
