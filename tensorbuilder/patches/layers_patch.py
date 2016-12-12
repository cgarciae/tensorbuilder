import tensorflow as tf
import inspect
import functools
from tensorflow.contrib import layers
from tensorflow.contrib.layers import fully_connected, convolution2d
from tensorbuilder import TensorBuilder
from phi import utils, P, patch
from phi.builder import Builder


class LayerBuilder(Builder):
    """docstring for LayerBuilder."""

    @property
    def TensorBuilder(self):
        return TensorBuilder()._unit(self._f, self._refs)

#Add property to TensorBuilder
TensorBuilder.layers = property(lambda self: LayerBuilder()._unit(self._f, self._refs))

# patch all layer functions
patch.builder_with_members_from_1(LayerBuilder, layers, module_alias="tf.contrib.layers") #, _return_type=TensorBuilder)

# fully conneted layers
blacklist = (
    ["relu_layer"] +
    TensorBuilder.__core__
)

funs = ( (name, f) for (name, f) in inspect.getmembers(tf.nn, inspect.isfunction) if name not in blacklist )

def register_layer_functions(name, f):
    explanation = """and the keyword argument `activation_fn` is set to `tf.nn.{0}`.""".format(name)

    @TensorBuilder.Register1("tf.contrib.layers", name + "_layer", wrapped=fully_connected, explanation=explanation) #, _return_type=TensorBuilder)
    def layer_function(*args, **kwargs):
        kwargs['activation_fn'] = f
        return fully_connected(*args, **kwargs)

def register_conv_layer_functions(name, f):
    explanation = """and the keyword argument `activation_fn` is set to `tf.nn.{0}`.""".format(name)

    @TensorBuilder.Register1("tf.contrib.layers", name + "_conv2d_layer", wrapped=convolution2d, explanation=explanation) #, _return_type=TensorBuilder)
    def layer_function(*args, **kwargs):
        kwargs['activation_fn'] = f
        return convolution2d(*args, **kwargs)

for name, f in funs:
    register_layer_functions(name, f)
    register_conv_layer_functions(name, f)



#linear_layer
explanation = """and the keyword argument `activation_fn` is set to `None`."""

@TensorBuilder.Register1("tf.contrib.layers", alias="linear_layer", wrapped=fully_connected, explanation=explanation) #, _return_type=TensorBuilder)
def linear_layer(*args, **kwargs):
    kwargs['activation_fn'] = None
    return tf.contrib.layers.fully_connected(*args, **kwargs)

@TensorBuilder.Register1("tf.contrib.layers", alias="linear_conv2d_layer", wrapped=convolution2d, explanation=explanation) #, _return_type=TensorBuilder)
def linear_conv2d_layer(*args, **kwargs):
    kwargs['activation_fn'] = None
    return tf.contrib.layers.fully_connected(*args, **kwargs)

def _polynomial(tensor):
    size = int(tensor.get_shape()[1])
    pows = [ tf.pow(tensor[:, n], n + 1) for n in range(size) ]
    return tf.transpose(tf.pack(pows))

explanation = """
However, it uses an activation function of the form
```
y(i) = z(i)^(i+1)
```
where `z = w*x + b`
"""

@TensorBuilder.Register1("tb", alias="polynomial_layer", wrapped=fully_connected, explanation=explanation) #, _return_type=TensorBuilder)
def polynomial_layer(*args, **kwargs):
    kwargs['activation_fn'] = _polynomial
    return layers.fully_connected(*args, **kwargs)


whitelist = ["convolution2d", "max_pool2d", "avg_pool2d", "flatten"]
patch.builder_with_members_from_1(TensorBuilder, layers, module_alias="tf.contrib.layers", whitelist=lambda x: x in whitelist) #, _return_type=TensorBuilder)
