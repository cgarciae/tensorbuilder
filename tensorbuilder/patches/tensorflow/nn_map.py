"""
All functions in this module are automatically generated. They help create custom layers and mappings for a Builder based on the functions in `tf.nn`. It works the following way:

* Let `f` be a function in `tf.nn`, then the functions with name `f_layer` and `map_f` exists in this module and take a Builder as its first argument. `f_layer` and `map_f` receive \*args and \*\*kwargs which are forwarded to `f`.
* `f_layer` functions connect a Builder to a layer with `f` as its activation function.
* `map_f` functions just map `f` over the tensor inside the Builder.


** Examples **

The following example show you how to 

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

for _f_name, f in inspect.getmembers(tf.nn, inspect.isfunction):
    tb.Builder.register_map_method(f, "tensorflow.nn")
