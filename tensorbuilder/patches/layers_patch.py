import tensorflow as tf
import inspect
import functools
from tensorflow.contrib import layers
from tensorflow.contrib.layers import fully_connected
from tensorbuilder import TensorBuilder
from tensorbuilder import Builder


class LayerBuilder(Builder):
    """docstring for LayerBuilder."""

LayerBuilder.__core__ = [ name for name, f in inspect.getmembers(LayerBuilder, predicate=inspect.ismethod) ]

#Add property to TensorBuilder
TensorBuilder.layers = property(lambda self: LayerBuilder(self.f))

# fully conneted layers
blacklist = (
    ["relu_layer", "device"] +
    TensorBuilder.__core__
)

funs = ( (name, f) for (name, f) in inspect.getmembers(tf.nn, inspect.isfunction) if name not in blacklist )

def register_layer_functions(name, f):
    explanation = """and the keyword argument `activation_fn` is set to `tf.nn.{0}`.""".format(name)

    @LayerBuilder.register("tf.contrib.layers", alias="{0}".format(name), wrapped=fully_connected, explanation=explanation, _return_type=TensorBuilder)
    def layer_function(*args, **kwargs):
        kwargs['activation_fn'] = f
        return tf.contrib.layers.fully_connected(*args, **kwargs)

for name, f in funs:
    register_layer_functions(name, f)


#linear_layer
explanation = """and the keyword argument `activation_fn` is set to `None`."""

@LayerBuilder.register("tf.contrib.layers", alias="linear", wrapped=fully_connected, explanation=explanation, _return_type=TensorBuilder)
def linear(*args, **kwargs):
    kwargs['activation_fn'] = None
    return tf.contrib.layers.fully_connected(*args, **kwargs)


print [ name for name, f in inspect.getmembers(layers, inspect.isfunction) ]
