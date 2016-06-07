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
		.map_dropout(keep_prob)
		.softmax_layer(3)
		.tensor
	)
"""

import tensorflow as tf
import tensorbuilder as tb
import inspect
from collections import namedtuple
import signature


for _nn_name, f in inspect.getmembers(tf.nn, inspect.isfunction):
 	_layer_name = _nn_name + "_layer"
 	_map_name = "map_" + _nn_name
 	_f_signature = signature.get_method_sig(f)
 	_f_docs = inspect.getdoc(f)


 	exec("""

@tb._immutable
def {1}(builder, size, *args, **kwargs):
	\"\"\"
THIS METHOD IS AUTOMATICALLY GENERATED

**@_immutable**

It connects the current tensor to a layer whos output function is `tf.nn.{0}`. The keyword parameters `weights_name`, `bias` and `bias_name` are set to defaults if they are not present in \*\*kwargs. Any additional positional (\*args) and keyword arguments (\*\*kwargs) will be forwarded to `tf.nn.{0}`.

**Note:**

Its expected that tf.nn.{0} can receive `builder.tensor` as a first parameter.

** TensorFlow documentation for `tf.nn.{0}`**

	def {2}

{3}

	\"\"\"
	name = {1} if "name" not in kwargs else kwargs["name"]
	weights_name = None if "weights_name" not in kwargs else kwargs["weights_name"]
	bias = True if "bias" not in kwargs else kwargs["bias"]
	bias_name = None if "bias_name" not in kwargs else kwargs["bias_name"]

	if "weights_name" in kwargs:
		del kwargs["weights_name"]
	if "bias" in kwargs:
		del kwargs["bias"]
	if "bias_name" in kwargs:
		del kwargs["bias_name"]

	return (
		builder
		.connect_layer(size, weights_name=weights_name, bias=bias, bias_name=bias_name)
		.map(tf.nn.{0}, *args, **kwargs)
	)

tb.Builder.{1} = {1}
tb.BuilderTree.{1} = {1}

 	""".format(_nn_name, _layer_name, _f_signature, _f_docs))

 	exec("""

@tb._immutable
def {1}(builder, *args, **kwargs):
	\"\"\"
THIS METHOD IS AUTOMATICALLY GENERATED

**@_immutable**

It maps `tf.nn.{0}` over the current tensor. All positional (\*args) and keyword arguments (\*\*kwargs) are forwarded to `tf.nn.{0}`.

**Note:**

Its expected that tf.nn.{0} can receive `builder.tensor` as a first parameter.

** TensorFlow documentation for `tf.nn.{0}` **

	def {2}

{3}
	\"\"\"
	return (
		builder
		.map(tf.nn.{0}, *args, **kwargs)
	)

tb.Builder.{1} = {1}

 	""".format(_nn_name, _map_name, _f_signature, _f_docs))
