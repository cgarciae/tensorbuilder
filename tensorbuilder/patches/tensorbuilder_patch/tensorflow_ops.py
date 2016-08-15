import tensorflow as tf

builders_blacklist = []

def patch_classes(Builder, BuilderTree, Applicative):
    Builder.register_map_method(tf.contrib.layers.flatten, "tf.contrib.layers")
    Builder.register_map_method(tf.contrib.layers.fully_connected, "tf.contrib.layers")
    Builder.register_map_method(tf.contrib.layers.convolution2d, "tf.contrib.layers")