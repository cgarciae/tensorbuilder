import tensorflow as tf
from tensorbuilder.core.builders import BuilderBase, BuilderTreeBase
from tensorbuilder.core.dsl import ApplicativeBase
from tensorbuilder.core import utils
import tflearn as tl
import inspect


def patch_classes(Builder, BuilderTree, Applicative):

    ###############################
    #### TREE
    ###############################
    def _tree_fully_connected(tree, size, *args, **kwargs):
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

    BuilderTree.register_method(_tree_fully_connected, "tensorbuilder.patches.tensorflow.fully_connected", alias="fully_connected")

    ###############################
    #### BUILDER
    ###############################

    #fully_connected
    Builder.register_map_method(tf.contrib.layers.fully_connected, "tf.contrib.layers")

    #convolution2d
    Builder.register_map_method(tf.contrib.layers.convolution2d, "tf.contrib.layers")

    ###############################
    # tf + tf.nn
    ###############################

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

    _banned = ["relu_layer"] + BuilderBase.__core__ + BuilderTreeBase.__core__

    _tf_funs = (
        [ (name, f, "tf.nn") for (name, f) in inspect.getmembers(tf.nn, inspect.isfunction) if name not in _banned ] +
        [ (name, f, "tf") for (name, f) in inspect.getmembers(tf, inspect.isfunction) if name not in _banned ]
    )


    for _name, f, _module_name in _tf_funs:
        _layer_name = _name + "_layer"
        _f_signature = utils.get_method_sig(f)
        _f_docs = inspect.getdoc(f)

        _layer_method = _get_layer_method(f)

        _layer_method.__name__ = _layer_name
        _layer_method.__doc__ = _f_docs

        # Builder
        Builder.register_map_method(f, _module_name) #This should go first
        Builder.register_method(_layer_method, _module_name, alias=_layer_name)


        # Tree
        BuilderTree.register_method(_layer_method, _module_name, alias=_layer_name)


    ###############################
    # tflearn
    ###############################

    Builder.register_map_method(tl.layers.conv.max_pool_2d, "tflearn.layers")

    #######################
    ### clone tb + tr
    #######################

    def _get_app_method(f):
        def _method(app, *args, **kwargs):
            def _lambda(builder):
                g = getattr(builder, f.__name__)
                return g(*args, **kwargs)

            return app.compose(_lambda)

        return _method

    _builder_excluded = ["copy", "compose"] + ApplicativeBase.__core__
    _tree_excluded = ["copy", "connect_layer", "compose"] + ApplicativeBase.__core__

    _dsl_funs = (
        [ ("BuilderTree", _name, f) for  _name, f in inspect.getmembers(BuilderTree, inspect.ismethod) if _name[0] != '_' and _name not in _tree_excluded ] +
        [ ("Builder", _name, f) for  _name, f in inspect.getmembers(Builder, inspect.ismethod) if _name[0] != '_' and _name not in _builder_excluded ]
    )

    for _module_name, _name, f in _dsl_funs:
        _f_signature = utils.get_method_sig(f)
        _f_docs = inspect.getdoc(f)

        _method = _get_app_method(f)

        _method.__name__ = _name
        _method.__doc__ = _f_docs

        Applicative.register_method(_method, _module_name)
