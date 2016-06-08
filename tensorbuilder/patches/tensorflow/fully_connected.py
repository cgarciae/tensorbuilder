import tensorflow as tf
import tensorbuilder as tb

def _add_builders(builders):
    tensor = None
    variables = {}

    for builder in builders:
        if tensor == None:
            tensor = builder.tensor
        else:
            tensor += builder.tensor

    return tb.Builder(tensor)

def _tree_fully_connected(tree, size, *args, **kwargs):
    activation_fn = None
    fun_args = ()
    fun_kwargs = {}

    if "fun_args" in kwargs:
        fun_args = kwargs["fun_args"]
        del kwargs["fun_args"]

    if "fun_kwargs" in kwargs:
        fun_kwargs = kwargs["fun_kwargs"]
        del kwargs["fun_kwargs"]

    if "activation_fn" in kwargs:
        activation_fn = kwargs["activation_fn"]
        del kwargs["activation_fn"]


    builders = ( builder.fully_connected(size, *args, **kwargs) for builder in tree )
    builder =  _add_builders(builders)

    if activation_fn:
        builder = builder.map(activation_fn, *fun_args, **fun_kwargs)

    return builder

tb.BuilderTree.register_method(_tree_fully_connected, "tensorbuilder.patches.tensorflow.fully_connected", alias="fully_connected")
tb.Builder.register_map_method(tf.contrib.layers.fully_connected, "tf.contrib.layers")
