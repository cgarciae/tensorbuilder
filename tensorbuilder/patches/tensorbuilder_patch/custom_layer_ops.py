import tensorflow as tf
from tensorbuilder.core.builders import BuilderBase, BuilderTreeBase
import inspect
from tensorbuilder.core import utils

builders_blacklist = (
    ["relu_layer"] +
    BuilderBase.__core__ +
    BuilderTreeBase.__core__
)

def patch_classes(Builder, BuilderTree, Applicative, builders_blacklist):

    #### TREE ####
    def _tree_fully_connected(tree, size, *args, **kwargs):
        """
        Reduces all leaf nodes of a to a single layer. To do this, it first creates a `fully_connected` linear layer of size `size` for each leaf node, then it adds all these together to create a single layer. At this point if `activation_fn` is defined it applies it to this sum.

        > **Note:** This function behaves slightly different to `tf.contrib.layers.fully_connected` since that function has `tf.nn.relu` as the default for `activation_fn`, that behavior might be unexpected so we initialize it as `None`.

        **Arguments**

        * `size`: the size of the resulting layer
        * All other \*args and \*\*kwargs are forwarded to `tf.contrib.layers.fully_connected`

        **Return**

        Builder

        **Examples**
        """
        activation_fn = None

        if "activation_fn" in kwargs:
            activation_fn = kwargs["activation_fn"]
            del kwargs["activation_fn"]

        builder = (
            tree
            .map_each(tf.contrib.layers.fully_connected, size, *args, **kwargs)
            .reduce(tf.add)
        )

        if activation_fn:
            builder = builder.map(activation_fn)

        return builder

    BuilderTree.register_method(
        _tree_fully_connected,
        "tensorbuilder.patches.tensorflow.fully_connected",
        alias = "fully_connected",
        doc = _tree_fully_connected.__doc__
    )

    #### BUILDER ####
    def _get_layer_method(f, name):
        def _layer_method(builder, size, *args, **kwargs):
            kwargs['activation_fn'] = f

            return builder.fully_connected(size, *args, **kwargs)

        return _layer_method


    _tf_funs = (
        [ (name, f, "tf.nn") for (name, f) in inspect.getmembers(tf.nn, inspect.isfunction) if name not in builders_blacklist ] +
        [ (name, f, "tf") for (name, f) in inspect.getmembers(tf, inspect.isfunction) if name not in builders_blacklist ]
    )


    for _name, f, _module_name in _tf_funs:
        _layer_name = _name + "_layer"
        _f_signature = utils.get_method_sig(f)
        _f_docs = inspect.getdoc(f)

        _layer_method = _get_layer_method(f, _layer_name)

        _layer_method.__name__ = _layer_name
        _layer_method.__doc__ = """
THIS METHOD IS AUTOMATICALLY GENERATED

Alias for `.fully_connected(size, activation_fn = {1}.{0}, ...)`

**Arguments**

* `size`: the size of the resulting layer
* All other \*args and \*\*kwargs are forwarded to `tf.contrib.layers.fully_connected`

**Return**

Builder

**Origial documentation for {1}.{0}**

    def {2}:

{3}
        """.format(_name, _module_name, _f_signature, _f_docs)

        # Builder
        Builder.register_map_method(f, _module_name) #This should go first
        Builder.register_method(_layer_method, _module_name, alias = _layer_name, doc = _layer_method.__doc__)


        # Tree
        BuilderTree.register_method(_layer_method, _module_name, alias=_layer_name, doc = _layer_method.__doc__)

    #######################
    ### linear_layer
    #######################

    def linear_layer(builder, size, *args, **kwargs):
        """
        Alias for `.fully_connected(size, activation_fn = None, ...)`

        **Arguments**

        * `size`: the size of the resulting layer
        * All other \*args and \*\*kwargs are forwarded to `tf.contrib.layers.fully_connected`

        **Return**

        Builder
        """
        kwargs['activation_fn'] = None

        return builder.fully_connected(size, *args, **kwargs)

    Builder.register_method(linear_layer, "tensorbuilder")
    BuilderTree.register_method(linear_layer, "tensorbuilder")
