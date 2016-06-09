"""
All functions in this module are automatically generated. They help create custom layers and mappings for a Builder based on the functions in `tf.nn`. It works the following way:

* Let `f` be a function in `tf.nn`, then the functions with name `f_layer` and `map_f` exists in this module and take a Builder as its first argument. `f_layer` and `map_f` receive \*args and \*\*kwargs which are forwarded to `f`.
* `f_layer` functions connect a Builder to a layer with `f` as its activation function.
* `map_f` functions just map `f` over the tensor inside the Builder.


** Examples **

import tensorflow as tf
import tensorbuilder as tb

x = tf.placeholder(tf.float32, shape=[None, 5])
keep_prob = tf.placeholder(tf.float32)

h = (
	x.builder()
	.tanh_layer(10)
	.dropout_map(keep_prob)
	.softmax_layer(3)
	.tensor
)
"""

import tensorflow as tf
import tensorbuilder as tb
import inspect
import tensorbuilder.utils as utils

def _get_layer_method(f):
    def _layer_method(builder, size, *args, **kwargs):
        fun_args = ()
        fun_kwargs = {}

        if "fun_args" in kwargs:
        	fun_args = kwargs["fun_args"]
        	del kwargs["fun_args"]

        if "fun_kwargs" in kwargs:
        	fun_kwargs = kwargs["fun_kwargs"]
        	del kwargs["fun_kwargs"]

        return (
            builder
            .fully_connected(size, *args, **kwargs)
            .map(f, *fun_args, **fun_kwargs)
        )
    return _layer_method

for _nn_name, f in inspect.getmembers(tf.nn, inspect.isfunction):
    _layer_name = _nn_name + "_layer"
    _f_signature = utils.get_method_sig(f)
    _f_docs = inspect.getdoc(f)

    _layer_method = _get_layer_method(f)

    _layer_method.__name__ = _nn_name
    _layer_method.__docs__ = _f_docs

    tb.Builder.register_method(_layer_method, "tensorflow.nn", alias=_layer_name)
    tb.BuilderTree.register_method(_layer_method, "tensorflow.nn", alias=_layer_name)
