import tensorflow as tf

builders_blacklist = [
    "dynamic_rnn", "rnn"
]

def dynamic_rnn(inputs, cell, *args, **kwargs):
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, *args, **kwargs)
    return outputs

def rnn(inputs, cell, *args, **kwargs):
    outputs, state = tf.nn.rnn(cell, inputs, *args, **kwargs)
    return outputs

def patch_classes(Builder, BuilderTree, Applicative):
    #tf.contrib.layers
    Builder.register_map_method(tf.contrib.layers.flatten, "tf.contrib.layers")
    Builder.register_map_method(tf.contrib.layers.fully_connected, "tf.contrib.layers")
    Builder.register_map_method(tf.contrib.layers.convolution2d, "tf.contrib.layers")

    #tf.nn
    Builder.register_map_method(dynamic_rnn, "tf.nn")
    Builder.register_map_method(rnn, "tf.nn")
