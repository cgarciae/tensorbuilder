"""
#TODO: List the core methods here. Explain the inclusion of the method from the `tensorbuilder.patch` module.
"""

import numpy as np
import functools
import utils
import inspect
from utils import immutable



class Builder(object):
    """
    The Builder class is a wrapper around a Tensor. Most of its method are immutable, that is, they don't modify the caller object but always return a new builder.

    To create a builder from a tensor you have these options:

    1. Use the `tensorbuilder.tensorbuilder.build` function

            tb.build(tensor)

    2. Use the monkey-patched method on the Tensor class

            tensor.builder()

    This class without patches only includes this basic methods:

    * `tensorbuilder.tensorbuilder.Builder.map`
    * `tensorbuilder.tensorbuilder.Builder.then`
    * `tensorbuilder.tensorbuilder.Builder.branch`

    It also includes the following static methods to register external functions as methods of this class. Library authors should use these to create patches:

    * `tensorbuilder.tensorbuilder.Builder.register_method`
    * `tensorbuilder.tensorbuilder.Builder.register_map_method`


    """
    def __init__(self, tensor):
        super(Builder, self).__init__()

        self._tensor = tensor
        """A `tensorflow` Tensor."""

    def tensor(self):
        "Returns the Tensor contianed by the Builder"
        return self._tensor

    def copy(self):
        """Returns a copy of this Builder"""
        return Builder(self._tensor)


    @staticmethod
    def register_method(fn, library_path, alias=None, doc=None):
        """
        This method enables you to register any function `fn` that takes a Builder as its first argument as a method of the Builder class.

        **Arguments**

        * `fn`: a function that atleast takes a Builder as its first argument.
        * `library_path`: the route of the librar from which this function was taken, used for documentation purposes.
        * `alias`: allows you to specify the name of the method, it will take the name of the function if its `None`.
        * `doc`: the documentation for the method, if `None` a predefied documentation will be generated based on the documentation of `fn`.

        **Return**

        `None`

        **Examples**

        In this example we will create a funtion and register it as a method called `relu_dropout_layer`

            import tensorflow as tf
            import tensorbuilder as tb


            def relu_dropout(builder, size, keep_prob):
                \"\"\"Fully connect to a relu layer of size `size` and apply dropout with `keep_prob`\"\"\"
                return (
                    builder.map(tf.contrib.layers.fully_connected, size)
                    .map(tf.nn.relu)
                    .map(tf.nn.dropout, keep_prob)
                )

            tb.Builder.register_method(relu_dropout_layer, "my.lib", alias="relu_dropout_layer")
        """

        fn_signature = utils.get_method_sig(fn)
     	fn_docs = inspect.getdoc(fn)
        original_name = fn.__name__
        name = alias if alias else original_name

        fn.__name__ = name
        fn.__doc__ = doc if doc else _builder_register_method_docs(original_name, library_path, name, fn_signature, fn_docs)

        setattr(Builder, name, fn)
        #exec("Builder.{0} = fn".format(name))


    @staticmethod
    def register_map_method(fn, library_path, alias=None, doc=None):
        """
        This method enables you to register any function `fn` that takes a Tensor as its first argument and returns a Tensor as a method of the Builder class. The resulting method is created by *lifting* the function to work with a Builder.

        **Arguments**

        * `fn`: a function of type `Tensor -> Tensor`.
        * `library_path`: the route of the librar from which this function was taken, used for documentation purposes.
        * `alias`: allows you to specify the name of the method, it will take the name of the function if its `None`.
        * `doc`: the documentation for the method, if `None` a predefied documentation will be generated based on the documentation of `fn`.

        **Return**

        `None`

        **Examples**

        In this example we will register `tf.reshape` as a method of the Builder class

            import tensorflow as tf
            import tensorbuilder as tb

            tb.Builder.register_map_method(tf.reshape, "tf")
        """
        fn_signature = utils.get_method_sig(fn)
     	fn_docs = inspect.getdoc(fn)
        original_name = fn.__name__
        name = alias if alias else original_name

        lifted = _lift(fn)
        lifted.__name__ = name
        lifted.__doc__ = doc if doc else _builder_register_map_method_docs(original_name, library_path, name, fn_signature, fn_docs)

        setattr(Builder, name, lifted)
        #exec("Builder.{0} = lifted".format(name))


    @immutable
    def map(builder, fn, *args, **kwargs):
        """
        `@immutable`

        Let **x** be Tensor inside a Builder `builder` and **fn** be a function from a tensor to a tensor, then `builder.map(fn, \*args, **kwargs)` computes `fn(x, *args, **kwargs) and stores the result inside a Builder`. While TensorBuilder promotes the use **patches** like `tensorbuilder.patch` to make the syntax nicer, the truth is that you could you can do a lot of things just using `map`, all you need is that you have a library or a set of custom functions that accept a tensor as its first argument.

        **Parameters**

        * `fn`: a function of type `tensor -> tensor`.
        * All extra positional and named arguments are forwarded to `fn`

        **Return**

        * `tensorbuilder.tensorbuilder.Builder`

        **Examples**

            import tensorflow as tf
            import tensorflow.contrib.layers
            import tensorbuilder as tb

            x = tf.placeholder(tf.float32, shape=[None, 40])
            keep_prob = tf.placeholder(tf.float32)

            h = (
            	x.builder()
            	.map(layers.fully_connected, 100, activation_fn=tf.nn.tanh)
            	.map(tf.nn.dropout, keep_prob)
            	.map(layers.fully_connected, 30, activation_fn=tf.nn.softmax)
            	.tensor()
            )

            print(h)

        """

        builder._tensor = fn(builder._tensor, *args, **kwargs)
        return builder

    @immutable
    def then(builder, fn, *args, **kwargs):
        """
        `@immutable`

        Expects a function **fn** with type `builder -> builder`. This method is used primarily to manupilate the Builder with very fine grain control through the fluent immutable API.

        **Parameters**

        * `fn`: a function of type `builder -> builder`.

        **Return**

        * `tensorbuilder.tensorbuilder.Builder`

        ** Example **

        """
        return fn(builder, *args, **kwargs)

    @immutable
    def branch(builder, fn):
        """
        `@immutable`

        Expects a function **fn** with type `Builder -> list( Builder | BuilderTree )`. This method enables you to *branch* the computational graph so you can easily create neural networks with more complex topologies. You can later

        **Parameters**

        * `fn`: a function of type `Builder -> list( Builder | BuilderTree )`.

        **Return**

        * `tensorbuilder.tensorbuilder.BuilderTree`

        ** Example **

        """
        return branches(fn(builder))

    @immutable
    def __iter__(builder):
        yield builder


class BuilderTree(object):
    """
    BuilderTree is a class that enables you to perform computations over a complex branched builder. It contains methods to handle the leaf `tensorbuilder.tensorbuilder.Builder` nodes.
    """
    def __init__(self, branches):
        super(BuilderTree, self).__init__()
        self.branches = branches
        """
        A list that can contain elements that are of type `tensorbuilder.tensorbuilder.Builder` or `tensorbuilder.tensorbuilder.BuilderTree`.
        """

    def copy(self):
        return BuilderTree(list(self.branches))

    @immutable
    def reduce(tree, fn, initializer=None):
        """
        `@immutable`

        Expects a function **fn** with type `(Tensor, Tensor) -> Tensor` and optionally an `initializer` and applies python [reduce](https://docs.python.org/2/library/functions.html#reduce) function to `tensorbuilder.tensorbuilder.BuilderTree.tensors` using these arguments; the resulting Tensor is the wrapped inside a Builder.

        **Parameters**

        * `fn`: a function of type `(Tensor, Tensor) -> Tensor`.
        * `initializer`: an optional Tensor as initial element of the folding operation (default: `None`)s

        **Return**

        * `tensorbuilder.tensorbuilder.Builder`

        ** Example **

        In this example we connect the whole tree to single softmax output layer of size 5, to do that we will separately map each leaf Tensor to a linear layer of size 5 and the add all the layers using reduce, finally we will apply a softmax function over the resulting layer.

            import tensorflow as tf
            import tensorbuilder as tb
            import tensorflow.contrib.layers as layers

            x = placeholder(tf.float32, shape=[None, 10])

            h = (
                x.builder()
                .branch(...) #perform some branching operation to obtain a BuilderTree
                .map_each(layers.fully_connected, 5)
                .reduce(tf.add)
                .map(tf.nn.softmax)
            )
        """
        if initializer != None:
            tensor = functools.reduce(fn, tree.tensors(), initializer)
        else:
            tensor = functools.reduce(fn, tree.tensors())

        return build(tensor)

    @immutable
    def map_each(tree, fn, *args, **kwargs):
        """
        `@immutable`

        Expects a function **fn** with type `Tensor -> Tensor` and applies this function to all leaf Tensors separately, resulting in a new BuilderTree.

        **Parameters**

        * `fn`: a function of type `Tensor -> Tensor`.
        * All additional \*args and \*\*kwargs are forwarded to `fn`

        **Return**

        * `tensorbuilder.tensorbuilder.BuilderTree`

        ** Example **

        In this example we will applay dropout to all leaf Tensors using `map_each`

            import tensorflow as tf
            import tensorbuilder as tb

            x = placeholder(tf.float32, shape=[None, 10])
            keep_prob = tf.placeholder(tf.float32)

            h = (
                x.builder()
                .branch(...) #perform some branching operation to obtain a BuilderTree
                .map_each(tf.nn.dropout, keep_prob)
            )
        """
        tree.branches = [ builder.map(fn, *args, **kwargs) for builder in tree ]
        return tree

    @immutable
    def extract(tree, fn, *args, **kwargs):
        """
        `@immutable`

        Expects a function **fn** with type `list( Tensor ) -> Tensor` and applies this function to `tensorbuilder.tensorbuilder.BuilderTree.tensors`, the resulting Tensor is wrapped in Builder. This function

        **Parameters**

        * `fn`: a function of type `list( Tensor ) -> Tensor`.
        * All additional \*args and \*\*kwargs are forwarded to `fn`

        **Return**

        * `tensorbuilder.tensorbuilder.Builder`

        ** Example **

        In this example we will applay dropout to all leaf Tensors using `map_each`

            import tensorflow as tf
            import tensorbuilder as tb

            x = placeholder(tf.float32, shape=[None, 10])
            keep_prob = tf.placeholder(tf.float32)

            h = (
                x.builder()
                .branch(...) #perform some branching operation to obtain a BuilderTree
                .map_each(tf.nn.dropout, keep_prob)
            )
        """
        tensor = fn(tree.tensors(), *args, **kwargs)
        return BuilderTree(builders)



    @staticmethod
    def register_method(fn, library_path, alias=None, doc=None):
        """
        This method enables you to register any function `fn` that takes a BuilderTree as its first argument as a method of the Builder class.

        **Arguments**

        * `fn`: a function that atleast takes a BuilderTree as its first argument.
        * `library_path`: the route of the librar from which this function was taken, used for documentation purposes.
        * `alias`: allows you to specify the name of the method, it will take the name of the function if its `None`.
        * `doc`: the documentation for the method, if `None` a predefied documentation will be generated based on the documentation of `fn`.

        **Return**

        `None`

        **Examples**

        In this example we will create the method `fully_connected` for the BuilderTree class

            import tensorflow as tf
            import tensorbuilder as tb


            def _tree_fully_connected(tree, size, *args, **kwargs):
                activation_fn = None

                if "activation_fn" in kwargs:
                    activation_fn = kwargs["activation_fn"]
                    del kwargs["activation_fn"]

                builder = (
                    tree.map_each(tf.contrib.layers.fully_connected, size, *args, **kwargs)
                    .reduce(tf.add)
                )

                if activation_fn:
                    builder = builder.map(activation_fn)

                return builder

            tb.BuilderTree.register_method(_tree_fully_connected, "tensorbuilder.patches.tensorflow.fully_connected", alias="fully_connected")
        """
        fn_signature = utils.get_method_sig(fn)
     	fn_docs = inspect.getdoc(fn)
        original_name = fn.__name__
        name = alias if alias else original_name

        fn.__name__ = name
        fn.__doc__ = doc if doc else _tree_register_method_docs(original_name, library_path, name, fn_signature, fn_docs)

        setattr(BuilderTree, name, fn)
        #exec("BuilderTree.{0} = fn".format(name))

    @staticmethod
    def register_reduce_method(fn, library_path, alias=None, doc=None):
        """
        This method enables you to register a function `fn` of type `(Tensor, Tensor) -> Tensor` as a method of the Builder class.

        **Arguments**

        * `fn`: a function of type `(Tensor, Tensor) -> Tensor`
        * `library_path`: the route of the librar from which this function was taken, used for documentation purposes.
        * `alias`: allows you to specify the name of the method, it will take the name of the function if its `None`.
        * `doc`: the documentation for the method, if `None` a predefied documentation will be generated based on the documentation of `fn`.

        **Return**

        `None`

        **Examples**

        In this example we will create the method `reduce_add` for the BuilderTree class

            import tensorflow as tf
            import tensorbuilder as tb

            tb.BuilderTree.register_reduce_method(tf.add, "tf", alias="reduce_add")
        """
        fn_signature = utils.get_method_sig(fn)
     	fn_docs = inspect.getdoc(fn)
        original_name = fn.__name__
        name = alias if alias else original_name

        _tree_method = _lift_tree_reduce(fn)

        _tree_method.__name__ = name
        _tree_method.__doc__ = doc if doc else _tree_register_reduce_method_docs(original_name, library_path, name, fn_signature, fn_docs)

        setattr(BuilderTree, name, _tree_method)
        #exec("BuilderTree.{0} = _tree_method".format(name))

    def builders(self):
        """
        Returns a flattened list `tensorbuilder.tensorbuilder.Builder`s contained by this tree. The whole result is flattened in case of sub-elements are also `tensorbuilder.tensorbuilder.BuilderTree`s.

        **Return**

        * `list( tensorbuilder.tensorbuilder.Builder )`

        ** Example **


        """
        return [ builder for builder in self ]

    def tensors(self):
        """
        Same as `tensorbuilder.tensorbuilder.BuilderTree.builders` but extracts the tensor from each `tensorbuilder.tensorbuilder.Builder`.

        **Return**

        * `list( tf.Tensor )`

        ** Example **

        """
        return [ builder._tensor for builder in self ]


    @immutable
    def __iter__(tree):
        """A generator function that lazily returns all the Builders contianed by this tree"""
        for branch in tree.branches:
            for builder in branch:
                yield builder



## Module Funs
def _tree_register_reduce_method_docs(original_name, library_path, name, fn_signature, fn_docs):
    return """
THIS METHOD IS AUTOMATICALLY GENERATED

**@immutable**

This method reduces the whole BuilderTree to a single Builder by applying `tensorbuilder.tensorbuilder.BuilderTree.reduce` with `{1}.{0}`.

** Original Documentation for `{1}.{0}`**

    def {3}

{4}
    """.format(original_name, library_path, name, fn_signature, fn_docs)

def _tree_register_method_docs(original_name, library_path, name, fn_signature, fn_docs):
    return """
THIS METHOD IS AUTOMATICALLY GENERATED

**@immutable**

This method the same as `{1}.{0}`.

** Original Documentation for `{1}.{0}`**

    def {3}

{4}
    """.format(original_name, library_path, name, fn_signature, fn_docs)


def _builder_register_method_docs(original_name, library_path, name, fn_signature, fn_docs):
    return """
THIS METHOD IS AUTOMATICALLY GENERATED

**@immutable**

This method the same as `{1}.{0}`.

** Original Documentation for `{1}.{0}`**

    def {3}

{4}
    """.format(original_name, library_path, name, fn_signature, fn_docs)


def _builder_register_map_method_docs(original_name, library_path, name, fn_signature, fn_docs):
    return """
THIS METHOD IS AUTOMATICALLY GENERATED

**@immutable**

This method is a lifted version the function `{1}.{0}` to work with `tensorbuilder.tensorbuilder.Builder`s. Instead of taking a Tensor as its first argument it takes a builder, the rest of the arguments are exactly the same.


** Original Documentation for `{1}.{0}`**

    def {3}

{4}
    """.format(original_name, library_path, name, fn_signature, fn_docs)


def _lift(fn):
    def _lifted(builder, *args, **kwargs):
        return builder.map(fn, *args, **kwargs)
    return _lifted

def _lift_tree_reduce(fn):
    def _tree_method(tree, *args, **kwargs):
        return tree.map_each(fn, *args, **kwargs)
    return _tree_method

def _map_partial(fn, *args, **kwargs):
    return lambda builder: builder.map(fn, *args, **kwargs)

def build(tensor):
    """
    Takes a tensor and returns a `tensorbuilder.tensorbuilder.Builder` that contians it. If function is also used to monkey-patch tensorflow's Tensor class with a method of the same name.

    ** Parameters **

    * `tensor`: a tensorflow Tensor

    #### Example

    The following example shows you how to construct a `tensorbuilder.tensorbuilder.Builder` from a tensorflow Tensor.

        import tensorflow as tf
        import tensorbuilder as tb

        a = tf.placeholder(tf.float32, shape=[None, 8])
        a_builder = tb.builder(a)

    The previous is the same as

        a_builder = tf.placeholder(tf.float32, shape=[None, 8]).builder()

    since tensorbuilder monkey-patches tensorflow's Tensor with this function as method.
    """
    return Builder(tensor)



def branches(builder_list):
    """
    Takes a list with elements of type `tensorbuilder.tensorbuilder.Builder` or `tensorbuilder.tensorbuilder.BuilderTree` and returns a `tensorbuilder.tensorbuilder.BuilderTree`

    ** Parameters **

    * `builder_list`: list of type `list( Builder | BuilderTree)`

    #### Example

    Given a list of Builders and/or BuilderTrees you construct a `tensorbuilder.tensorbuilder.BuilderTree` like this

        import tensorflow as tf
        import tensorbuilder as tb

        a = tf.placeholder(tf.float32, shape=[None, 8]).builder()
        b = tf.placeholder(tf.float32, shape=[None, 8]).builder()

        tree = tb.branches([a, b])

    `tensorbuilder.tensorbuilder.BuilderTree`s are usually constructed using `tensorbuilder.tensorbuilder.Builder.branch` of the `tensorbuilder.tensorbuilder.Builder` class, but you can use this for special cases

    """
    return BuilderTree(builder_list)
