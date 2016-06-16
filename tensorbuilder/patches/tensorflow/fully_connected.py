import tensorflow as tf
import tensorbuilder as tb


def _tree_fully_connected(tree, size, *args, **kwargs):
    activation_fn = None

    if "activation_fn" in kwargs:
        activation_fn = kwargs["activation_fn"]
        del kwargs["activation_fn"]

    builder = (
        tree
        .map_each(tf.contrib.layers.fully_connected, size, *args, **kwargs)
        .reduce(tf.add)
    )

    if activation_fn:
        builder = builder.map(activation_fn)

    return builder

tb.BuilderTree.register_method(_tree_fully_connected, "tensorbuilder.patches.tensorflow.fully_connected", alias="fully_connected")

tb.Builder.register_map_method(tf.contrib.layers.fully_connected, "tf.contrib.layers")
tb.Builder.register_map_method(tf.contrib.layers.convolution2d, "tf.contrib.layers")
