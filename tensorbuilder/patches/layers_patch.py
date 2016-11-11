import tensorflow as tf
import inspect
import functools
from tensorflow.contrib.layers import fully_connected



def patch(TensorBuilder):

    def register_layer_functions(name, f):
        explanation = """and the keyword argument `activation_fn` is set to `tf.nn.{0}`.""".format(name)

        @TensorBuilder.register("tf.contrib.layers", alias="{0}_layer".format(name), wrapped=fully_connected, explanation=explanation)
        def layer_function(*args, **kwargs):
            kwargs['activation_fn'] = f
            return tf.contrib.layers.fully_connected(*args, **kwargs)


    blacklist = (
        ["relu_layer", "device"] +
        TensorBuilder.__core__
    )

    funs = ( (name, f) for (name, f) in inspect.getmembers(tf.nn, inspect.isfunction) if name not in blacklist )

    for name, f in funs:
        register_layer_functions(name, f)


    #linear_layer
    explanation = """and the keyword argument `activation_fn` is set to `None`."""

    @TensorBuilder.register("tf.contrib.layers", alias="linear_layer", wrapped=fully_connected, explanation=explanation)
    def linear_layer(*args, **kwargs):
        kwargs['activation_fn'] = None
        return tf.contrib.layers.fully_connected(*args, **kwargs)
