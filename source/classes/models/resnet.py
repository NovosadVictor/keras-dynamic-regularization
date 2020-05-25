from tensorflow.keras.layers import \
    Input, \
    AveragePooling2D, \
    Dense, \
    Conv2D, \
    Flatten, \
    ReLU, \
    BatchNormalization, \
    Add
from tensorflow.keras.models import Model
from tensorflow import Tensor


class ResNet:
    @staticmethod
    def build(dropout, input_shape: list, n_classes: int) -> Model:
        inputs = Input(shape=input_shape)
        num_filters = 64

        t = BatchNormalization()(inputs)
        t = Conv2D(
            kernel_size=3,
            strides=1,
            filters=num_filters,
            padding='same',
        )(t)
        t = ResNet.__relu_with_normalization(t)

        num_blocks_list = [2, 5, 5, 2]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                t = ResNet.__residual_block(t, down_sample=(j == 0 and i != 0), filters=num_filters)
                t = dropout(t)
            num_filters *= 2

        t = AveragePooling2D(4)(t)
        t = Flatten()(t)
        outputs = Dense(n_classes, activation='softmax')(t)

        return Model(inputs, outputs)

    @staticmethod
    def __relu_with_normalization(inputs: Tensor) -> Tensor:
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)

        return bn

    @staticmethod
    def __residual_block(x: Tensor, down_sample: bool, filters: int, kernel_size: int = 3) -> Tensor:
        y = Conv2D(
            kernel_size=kernel_size,
            strides=(1 if not down_sample else 2),
            filters=filters,
            padding='same',
        )(x)
        y = ResNet.__relu_with_normalization(y)
        y = Conv2D(
            kernel_size=kernel_size,
            strides=1,
            filters=filters,
            padding='same',
        )(y)

        if down_sample:
            x = Conv2D(
                kernel_size=1,
                strides=2,
                filters=filters,
                padding='same',
            )(x)
        out = Add()([x, y])
        out = ResNet.__relu_with_normalization(out)

        return out
