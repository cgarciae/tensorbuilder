"""
All functions in this module are automatically generated. They help create custom layers and mappings for a Builder based on the functions in `tf.nn`. It works the following way:

* Let `f` be a function in `tf.nn`, then the functions with name `f_layer` and `map_f` exists in this module and take a Builder as its first argument. `f_layer` and `map_f` receive \*args and \*\*kwargs which are forwarded to `f`.
* `f_layer` functions connect a Builder to a layer with `f` as its activation function.
* `map_f` functions just map `f` over the tensor inside the Builder.
*

Calling `tensorbuilder.builder_nn.patch` monkey-patches all the functions in this module as methods of the Builder and BuilderTree classes.

** Examples **

	import tensorflow as tf
	import tensorbuilder as tb

	tb.builder_nn.patch()

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

_patches = []

def patch():
	"""
	Moneky-patches all functions in this modules as methods on the Builder class.
	"""
	for _patch in _patches:
		exec(_patch)

DefaultArgSpec = namedtuple('DefaultArgSpec', 'has_default default_value')

def _get_default_arg(args, defaults, arg_index):
    """ Method that determines if an argument has default value or not,
    and if yes what is the default value for the argument

    :param args: array of arguments, eg: ['first_arg', 'second_arg', 'third_arg']
    :param defaults: array of default values, eg: (42, 'something')
    :param arg_index: index of the argument in the argument array for which,
    this function checks if a default value exists or not. And if default value
    exists it would return the default value. Example argument: 1
    :return: Tuple of whether there is a default or not, and if yes the default
    value, eg: for index 2 i.e. for "second_arg" this function returns (True, 42)
    """
    if not defaults:
        return DefaultArgSpec(False, None)

    args_with_no_defaults = len(args) - len(defaults)

    if arg_index < args_with_no_defaults:
        return DefaultArgSpec(False, None)
    else:
        value = defaults[arg_index - args_with_no_defaults]
        if (type(value) is str):
            value = '"%s"' % value
        return DefaultArgSpec(True, value)

def _get_method_sig(method):
    """ Given a function, it returns a string that pretty much looks how the
    function signature would be written in python.

    :param method: a python method
    :return: A string similar describing the pythong method signature.
    eg: "my_method(first_argArg, second_arg=42, third_arg='something')"
    """

    # The return value of ArgSpec is a bit weird, as the list of arguments and
    # list of defaults are returned in separate array.
    # eg: ArgSpec(args=['first_arg', 'second_arg', 'third_arg'],
    # varargs=None, keywords=None, defaults=(42, 'something'))
    argspec = inspect.getargspec(method)
    arg_index=0
    args = []

    # Use the args and defaults array returned by argspec and find out
    # which arguments has default
    for arg in argspec.args:
        default_arg = _get_default_arg(argspec.args, argspec.defaults, arg_index)
        if default_arg.has_default:
            args.append("%s=%s" % (arg, default_arg.default_value))
        else:
            args.append(arg)
        arg_index += 1
    return "%s(%s)" % (method.__name__, ", ".join(args))

for _nn_name, f in inspect.getmembers(tf.nn, inspect.isfunction):
 	_layer_name = _nn_name + "_layer"
 	_map_name = "map_" + _nn_name
 	_f_signature = _get_method_sig(f)
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

_patches.append(\"tb.Builder.{1} = {1}\")
_patches.append(\"tb.BuilderTree.{1} = {1}\")

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

_patches.append(\"tb.Builder.{1} = {1}\")

 	""".format(_nn_name, _map_name, _f_signature, _f_docs))
