from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import inspect
import functools
from tensorflow.contrib import layers
from tensorflow.contrib.layers import fully_connected, convolution2d
from tensorbuilder import TensorBuilder
from phi import utils, P, Builder

blacklist = ["relu_layer"]

#########################
## LayerBuilder
#########################

class LayerBuilder(Builder):
    """docstring for LayerBuilder."""

    @property
    def TensorBuilder(self):
        return self >> TensorBuilder()

#Add property to TensorBuilder
TensorBuilder.layers = property(lambda self: LayerBuilder(self._f))

# patch all layer functions
LayerBuilder.PatchAt(1, layers, module_alias="tf.contrib.layers", return_type_predicate=TensorBuilder)

#########################
## Layer methods
#########################


whitelist = ["convolution2d", "max_pool2d", "avg_pool2d", "flatten"]
TensorBuilder.PatchAt(1, layers, module_alias="tf.contrib.layers", whitelist_predicate=whitelist)

assert TensorBuilder.convolution2d


#########################
## Afine layers
#########################

def afine_layer_wrapper(f):

    def g(*args, **kwargs):
        kwargs['activation_fn'] = f
        return fully_connected(*args, **kwargs)

    return g

TensorBuilder.PatchAt(1, tf.nn,
    method_wrapper=afine_layer_wrapper,
    method_name_modifier = "{0}_layer".format,
    blacklist_predicate=blacklist,
    explanation="""and the keyword argument `activation_fn` is set to `tf.nn.{original_name}`."""
)


#########################
## Convolutional layers
#########################

def convolution_layer_wrapper(f):

    def g(*args, **kwargs):
        kwargs['activation_fn'] = f
        return convolution2d(*args, **kwargs)

    return g

TensorBuilder.PatchAt(1, tf.nn,
    method_wrapper=convolution_layer_wrapper,
    method_name_modifier = "{0}_conv2d_layer".format,
    blacklist_predicate=blacklist,
    explanation="""and the keyword argument `activation_fn` is set to `tf.nn.{original_name}`."""
)

#########################
## Custom Methods
#########################

explanation = """and the keyword argument `activation_fn` is set to `None`."""

@TensorBuilder.Register("tf.contrib.layers", alias="linear_layer", wrapped=fully_connected, explanation=explanation) #, _return_type=TensorBuilder)
def linear_layer(*args, **kwargs):
    kwargs['activation_fn'] = None
    return tf.contrib.layers.fully_connected(*args, **kwargs)

@TensorBuilder.Register("tf.contrib.layers", alias="linear_conv2d_layer", wrapped=convolution2d, explanation=explanation) #, _return_type=TensorBuilder)
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

@TensorBuilder.Register("tensorbuilder", alias="polynomial_layer", wrapped=fully_connected, explanation=explanation) #, _return_type=TensorBuilder)
def polynomial_layer(*args, **kwargs):
    kwargs['activation_fn'] = _polynomial
    return layers.fully_connected(*args, **kwargs)


