import tensorflow as tf
from source.classes.dropout import DynamicDropout

dropout = DynamicDropout(0.5)


def grad(x):
    with tf.GradientTape() as tape:
        tape.watch(dropout.rate)
        out = dropout(x, training=True)
    return tape.gradient(out, dropout.rate)


print(grad(tf.constant([[3.]])))
print(grad(tf.constant([[2.]])))
print(grad(tf.constant([[5., 5.]])))
