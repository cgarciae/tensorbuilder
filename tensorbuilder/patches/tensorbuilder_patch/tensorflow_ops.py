import tensorflow as tf

builders_blacklist = [
    "dynamic_rnn", "rnn"
]

def dynamic_rnn(inputs, cell, *args, **kwargs):
    state_builder = None

    if 'state_builder' in kwargs:
        state_builder = kwargs['state_builder']
        del kwargs['state_builder']

    outputs, state = tf.nn.dynamic_rnn(cell, inputs, *args, **kwargs)

    if state_builder is not None:
        state_builder._tensor = state

    return outputs

def rnn(inputs, cell, *args, **kwargs):
    state_builder = None

    if 'state_builder' in kwargs:
        state_builder = kwargs['state_builder']
        del kwargs['state_builder']

    outputs, state = tf.nn.rnn(cell, inputs, *args, **kwargs)

    if state_builder is not None:
        state_builder._tensor = state

    return outputs

def patch_classes(Builder, BuilderTree, Applicative):
    #tf.contrib.layers
    Builder.register_map_method(tf.contrib.layers.flatten, "tf.contrib.layers")
    Builder.register_map_method(tf.contrib.layers.fully_connected, "tf.contrib.layers")
    Builder.register_map_method(tf.contrib.layers.convolution2d, "tf.contrib.layers")

    #tf.nn
    Builder.register_map_method(dynamic_rnn, "tf.nn")
    Builder.register_map_method(rnn, "tf.nn")
