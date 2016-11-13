import tensorflow as tf
import inspect
import functools
from tensorflow.contrib import layers
from tensorflow.contrib.layers import fully_connected
from tensorbuilder import TensorBuilder
from tensorbuilder import Builder
from tensorbuilder.builder import utils


class LayerBuilder(Builder):
    """docstring for LayerBuilder."""


#Add property to TensorBuilder
TensorBuilder.layers = property(lambda self: LayerBuilder(self.f))

# patch all layer functions
utils.patch_with_members_from_1(LayerBuilder, layers, module_alias="tf.contrib.layers")

# fully conneted layers
blacklist = (
    ["relu_layer", "device"] +
    TensorBuilder.__core__
)

funs = ( (name, f) for (name, f) in inspect.getmembers(tf.nn, inspect.isfunction) if name not in blacklist )

def register_layer_functions(name, f):
    explanation = """and the keyword argument `activation_fn` is set to `tf.nn.{0}`.""".format(name)

    @LayerBuilder.register_1("tf.contrib.layers", name, wrapped=fully_connected, explanation=explanation, _return_type=TensorBuilder)
    def layer_function(*args, **kwargs):
        kwargs['activation_fn'] = f
        return tf.contrib.layers.fully_connected(*args, **kwargs)

for name, f in funs:
    register_layer_functions(name, f)


#linear_layer
explanation = """and the keyword argument `activation_fn` is set to `None`."""

@LayerBuilder.register_1("tf.contrib.layers", alias="linear", wrapped=fully_connected, explanation=explanation, _return_type=TensorBuilder)
def linear(*args, **kwargs):
    kwargs['activation_fn'] = None
    return tf.contrib.layers.fully_connected(*args, **kwargs)
