"""
#TODO: List the core methods here. Explain the inclusion of the method from the `tensorbuilder.patch` module.
"""

import numpy as np
import functools
import utils
import inspect
from utils import immutable
from copy import deepcopy, copy
from types import MethodType
import sys
from abc import ABCMeta, abstractmethod

_count = [0]

class BuilderBase(object):
    """
    The Builder class is a wrapper around a Tensor. Most of its method are immutable, that is, they don't modify the caller object but always return a new builder.

    This class is a Functor because it has the `map` method, to be a Monad is only missing the `bind` method which is trivial to implement. This means that even if its expected that the inner element contained within a Builder is a Tensor, it can actually contian anything and you may use this to your advantage.


    """
    __metaclass__ = ABCMeta

    def __init__(self, tensor):
        super(BuilderBase, self).__init__()

        self._tensor = tensor
        """A `tensorflow` Tensor."""

    @abstractmethod
    def BuilderTree(self, builder_iterable):
        pass

    def tensor(self):
        "Returns the Tensor contianed by the Builder"
        return self._tensor

    def copy(self):
        """Returns a copy of this Builder"""
        return self._unit(self.tensor())

    def _unit(self, tensor):
        return self.__class__(tensor)

    def store_on(builder, other):
        other._tensor = builder.tensor()
        return builder

    @classmethod
    def register_method(cls, fn, library_path, alias=None, doc=None):
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
            from tensorbuilder import tb


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

        setattr(cls, name, fn)
        #exec("Builder.{0} = fn".format(name))


    @classmethod
    def register_map_method(cls, fn, library_path, alias=None, doc=None):
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
            from tensorbuilder import tb

            tb.Builder.register_map_method(tf.reshape, "tf")
        """
        fn_signature = utils.get_method_sig(fn)
     	fn_docs = inspect.getdoc(fn)
        original_name = fn.__name__
        name = alias if alias else original_name

        lifted = _lift(fn)
        lifted.__name__ = name
        lifted.__doc__ = doc if doc else _builder_register_map_method_docs(original_name, library_path, name, fn_signature, fn_docs)

        setattr(cls, name, lifted)


    @immutable
    def map(builder, fn, *args, **kwargs):
        """
        `@immutable`

        Let **x** be Tensor inside a Builder `builder` and **fn** be a function from a tensor to a tensor, then `builder.map(fn, \*args, **kwargs)` computes `fn(x, *args, **kwargs) and stores the result inside a Builder`. The Builder class comes with a lot of **patched** methods that help you do things quickly and make the syntax nicer, but if we don't have the method you need just pass the function you want to use to `map`, or even consider using `tensorbuilder.core.builders.Builder.register_map_method`.

        **Parameters**

        * `fn`: a function of type `tensor -> tensor`.
        * All extra positional and named arguments are forwarded to `fn`

        **Return**

        * `tensorbuilder.core.builders.Builder`

        **Examples**

            import tensorflow as tf
            from tensorflow.contrib import layers
            from tensorbuilder import tb

            x = tf.placeholder(tf.float32, shape=[None, 40])
            keep_prob = tf.placeholder(tf.float32)

            h = (
            	tb.build(x)
            	.map(layers.fully_connected, 100, activation_fn=tf.nn.tanh)
            	.map(tf.nn.dropout, keep_prob)
            	.map(layers.fully_connected, 30, activation_fn=tf.nn.softmax)
            	.tensor()
            )

            print(h)

        Same using the DSL

            import tensorflow as tf
            from tensorflow.contrib import layers
            from tensorbuilder import tb


            x = tf.placeholder(tf.float32, shape=[None, 40])
            keep_prob = tf.placeholder(tf.float32)

            h = tb.pipe(
            	x,
            	tb.map(layers.fully_connected, 100, activation_fn=tf.nn.tanh)
            	.map(tf.nn.dropout, keep_prob)
            	.map(layers.fully_connected, 30, activation_fn=tf.nn.softmax)
            	.tensor()
            )

            print(h)

        """
        tensor = fn(builder.tensor(), *args, **kwargs)
        return builder._unit(tensor)

    @immutable
    def then(builder, fn, *args, **kwargs):
        """
        `@immutable`

        Expects a function **fn** with type `builder -> builder`. This method is used primarily to manupilate the Builder with very fine grain control through the fluent immutable API.

        **Parameters**

        * `fn`: a function of type `builder -> builder`.

        **Return**

        * `tensorbuilder.core.builders.Builder`

        ** Example **

        """
        return fn(builder, *args, **kwargs)

    @immutable
    def branch(builder, fn):
        """
        `@immutable`

        Expects a function **fn** with type `Builder -> iterable( Builder | BuilderTree )`. This method enables you to *branch* the computational graph so you can easily create neural networks with more complex topologies.

        **Parameters**

        * `fn`: a function of type `Builder -> iterable( Builder | BuilderTree )`.

        **Return**

        * `tensorbuilder.core.builders.BuilderTree`

        ** Examples **

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = (
                tb.build(x)
                .branch(lambda x: [
                    x.relu_layer(20)
                ,
                    x.sigmoid_layer(20)
                ,
                    x.tanh_layer(20)
                ])
                .softmax_layer(5)
                .tensor()
            )

        Same with the DSL

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = tb.pipe(
                x,
                [
                    tb.relu_layer(20)
                ,
                    tb.sigmoid_layer(20)
                ,
                    tb.tanh_layer(20)
                ],
                tb.softmax_layer(5)
                .tensor()
            )

        """
        return builder.BuilderTree(fn(builder))

    @immutable
    def then_with(builder, scope_fn, *args, **kwargs):
        """
        `@immutable`

        Expects a function **fn** with that returns a "Disposable" (implement `__enter__` and `__exit__`) plus some \*args and \*\*kwargs, and return a function `g` that expects a function `h` of type `Builder -> Builder` such that

            .then_with(fn, *args, **kwargs)(h)

        roughly perform this computations (given the current `builder`)

            with fn(*args, **kwargs):
                return h(builder)

        For a more practical understanding look at the example.

        **Parameters**

        * `fn`: a function of type `Builder -> Disposable`.

        **Return**

        * Function of type `(Builder -> Builder)`

        ** Examples **

        Create a network with 3 branches and execute each on the devices "/gpu:0", "/gpu:1", "cpu:3" respectively

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = (
                tb.build(x)
                .branch(lambda x: [
                    x.then_with(tf.device, "/gpu:0")(lambda x:
                        x.relu_layer(20)
                        .linear_layer(5)
                    )
                ,
                    x.then_with(tf.device, "/gpu:1")(lambda x:
                        x.sigmoid_layer(20)
                        .linear_layer(5)
                    )
                ,
                    x.then_with(tf.device, "/cpu:0")(lambda x:
                        x.tanh_layer(20)
                        .linear_layer(5)
                    )
                ])
                .reduce(tf.add)
                .softmax()
                .tensor()
            )

        This looks much better with the DSL thanks to its support for scopes

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = tb.pipe(
                x,
                [
                    { tf.device("/gpu:0"):
                        tb.relu_layer(20)
                        .linear_layer(5)
                    }
                ,
                    { tf.device("/gpu:1"):
                        tb.sigmoid_layer(20)
                        .linear_layer(5)
                    }
                ,
                    { tf.device("/cpu:0"):
                        tb.tanh_layer(20)
                        .linear_layer(5)
                    }
                ],
                tb.reduce(tf.add)
                .softmax()
                .tensor()
            )

        """
        def _lambda(fn):
            with scope_fn(*args, **kwargs):
                y = fn(builder)
            return y
        return _lambda

    @immutable
    def __iter__(builder):
        yield builder


class BuilderTreeBase(object):
    """
    BuilderTree is a class that enables you to perform computations over a complex branched builder. It contains methods to handle the leaf `tensorbuilder.core.builders.Builder` nodes.
    """

    __metaclass__ = ABCMeta

    def __init__(self, builder_iterable):
        super(BuilderTreeBase, self).__init__()

        self._branches = list(builder_iterable)
        """
        An iterable that can contain elements that are of type `tensorbuilder.core.builders.Builder` or `tensorbuilder.core.builders.BuilderTree`.
        """

    @abstractmethod
    def Builder(self, tensor):
        pass

    def copy(self):
        return self._unit(self._branches)

    def _unit(self, branches):
        return self.__class__(branches)



    @immutable
    def reduce(tree, fn, initializer=None):
        """
        `@immutable`

        Expects a function **fn** with type `(Tensor, Tensor) -> Tensor` and optionally an `initializer` and applies python [reduce](https://docs.python.org/2/library/functions.html#reduce) function to `tensorbuilder.core.builders.BuilderTree.tensors` using these arguments; the resulting Tensor is the wrapped inside a Builder.

        **Parameters**

        * `fn`: a function of type `(Tensor, Tensor) -> Tensor`.
        * `initializer`: an optional Tensor as initial element of the folding operation (default: `None`)s

        **Return**

        * `tensorbuilder.core.builders.Builder`

        ** Example **

        Lets reduce the example on `tensorbuilder.core.builders.Builder.branch` this time doing the reduction ourselves instead of relying on the `*_layer` of `tensorbuilder.core.builders.BuilderTree` that do this for us

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = (
                tb.build(x)
                .branch(lambda x: [
                    x.relu_layer(20)
                    .linear_layer(5)
                ,
                    x.sigmoid_layer(20)
                    .linear_layer(5)
                ,
                    x.tanh_layer(20)
                    .linear_layer(5)
                ])
                .reduce(tf.add)
                .softmax()
                .tensor()
            )

        Same example using the DSL

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = tb.pipe(
                x,
                [
                    tb.relu_layer(20)
                    .linear_layer(5)
                ,
                    tb.sigmoid_layer(20)
                    .linear_layer(5)
                ,
                    tb.tanh_layer(20)
                    .linear_layer(5)
                ],
                tb.reduce(tf.add)
                .softmax()
                .tensor()
            )
        """
        if initializer != None:
            tensor = functools.reduce(fn, tree.tensors(), initializer)
        else:
            tensor = functools.reduce(fn, tree.tensors())

        return tree.Builder(tensor)

    @immutable
    def map_each(tree, fn, *args, **kwargs):
        """
        `@immutable`

        Expects a function **fn** with type `Tensor -> Tensor` and applies this function to all leaf Tensors separately, resulting in a new BuilderTree.

        **Parameters**

        * `fn`: a function of type `Tensor -> Tensor`.
        * All additional \*args and \*\*kwargs are forwarded to `fn`

        **Return**

        * `tensorbuilder.core.builders.BuilderTree`

        ** Example **

        Lets redu the example in `tensorbuilder.core.builders.BuilderTree.reduce` using `map_each` to reduce some code

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = (
                tb.build(x)
                .branch(lambda x: [
                    x.relu_layer(20)
                ,
                    x.sigmoid_layer(20)
                ,
                    x.tanh_layer(20)
                ])
                .map_each(tf.contrib.layers.fully_connected, 5, activation_fn=None)
                .reduce(tf.add)
                .softmax()
                .tensor()
            )

        Remember that this

            .map_each(tf.contrib.layers.fully_connected, 5, activation_fn=None)
            .reduce(tf.add)
            .softmax()

        is equivalent to just

            .softmax_layer(5)

        for `BuilderTree`s. Same example using the DSL

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = tb.pipe(
                x,
                [
                    x.relu_layer(20)
                ,
                    x.sigmoid_layer(20)
                ,
                    x.tanh_layer(20)
                ],
                tb.map_each(tf.contrib.layers.fully_connected, 5, activation_fn=None)
                .reduce(tf.add)
                .softmax()
                .tensor()
            )
        """
        branches = [ builder.map(fn, *args, **kwargs) for builder in tree ]
        return tree._unit(branches)

    @immutable
    def extract(tree, fn, *args, **kwargs):
        """
        `@immutable`

        Expects a function **fn** with type `list( Tensor ) -> Tensor` and applies this function to `tensorbuilder.core.builders.BuilderTree.tensors`, the resulting Tensor is wrapped in Builder. This function

        **Parameters**

        * `fn`: a function of type `list( Tensor ) -> Tensor`.
        * All additional \*args and \*\*kwargs are forwarded to `fn`

        **Return**

        * `tensorbuilder.core.builders.Builder`

        ** Example **

        Lets redu the example in `tensorbuilder.core.builders.BuilderTree.map_each` using `extract`

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = (
                tb.build(x)
                .branch(lambda x: [
                    x.relu_layer(20)
                ,
                    x.sigmoid_layer(20)
                ,
                    x.tanh_layer(20)
                ])
                .map_each(tf.contrib.layers.fully_connected, 5, activation_fn=None)
                .extract(lambda tensors: tf.add_n(tensors)) #or just .extract(tf.add_n)
                .softmax()
                .tensor()
            )

        Same example using the DSL

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = (
                x,
                [
                    tb.relu_layer(20)
                ,
                    tb.sigmoid_layer(20)
                ,
                    tb.tanh_layer(20)
                ],
                tb.map_each(tf.contrib.layers.fully_connected, 5, activation_fn=None)
                .extract(lambda tensors: tf.add_n(tensors)) #or just .extract(tf.add_n)
                .softmax()
                .tensor()
            )
        """
        tensor = fn(tree.tensors(), *args, **kwargs)
        return tree.Builder(tensor)



    @classmethod
    def register_method(cls, fn, library_path, alias=None, doc=None):
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
            from tensorbuilder import tb

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

        setattr(cls, name, fn)

    @classmethod
    def register_reduce_method(cls, fn, library_path, alias=None, doc=None):
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
            from tensorbuilder import tb

            tb.BuilderTree.register_reduce_method(tf.add, "tf", alias="reduce_add")
        """
        fn_signature = utils.get_method_sig(fn)
     	fn_docs = inspect.getdoc(fn)
        original_name = fn.__name__
        name = alias if alias else original_name

        _tree_method = _lift_tree_reduce(fn)

        _tree_method.__name__ = name
        _tree_method.__doc__ = doc if doc else _tree_register_reduce_method_docs(original_name, library_path, name, fn_signature, fn_docs)

        setattr(cls, name, _tree_method)


    def builders(self):
        """
        Returns a flattened list `tensorbuilder.core.builders.Builder`s contained by this tree. The whole result is flattened in case of sub-elements are also `tensorbuilder.core.builders.BuilderTree`s.

        **Return**

        * `list( tensorbuilder.core.builders.Builder )`

        ** Examples **

        This examples creates a network to that solves the XOR problem using sigmoid units

            import tensorflow as tf
            from tensorbuilder import tb

            x = tf.placeholder(tf.float32, shape=[None, 2])
            y = tf.placeholder(tf.float32, shape=[None, 1])


            #Network
            [activation_builder, trainer_builder] = (
                tb.build(x)

                .sigmoid_layer(2)
                .linear_layer(1)

                .branch(lambda logit:
                [
                    logit.sigmoid() # activation
                ,
                    logit
                    .sigmoid_cross_entropy_with_logits(y) # loss
                    .map(tf.train.AdamOptimizer(0.01).minimize) # trainer
                ])
                .builders()
            )

        Same example using the DSL

            import tensorflow as tf
            from tensorbuilder import tb

            x = tf.placeholder(tf.float32, shape=[None, 2])
            y = tf.placeholder(tf.float32, shape=[None, 1])


            #Network
            [activation_builder, trainer_builder] = tb.pipe(
                x,
                tb.sigmoid_layer(2)
                .linear_layer(1),
                [
                    tb.sigmoid() # activation
                ,
                    tb
                    .sigmoid_cross_entropy_with_logits(y) # loss
                    .map(tf.train.AdamOptimizer(0.01).minimize) # trainer
                ],
                tb.builders()
            )

        """
        return [ builder for builder in self ]

    def tensors(self):
        """
        Same as `tensorbuilder.core.builders.BuilderTree.builders` but extracts the tensor from each `tensorbuilder.core.builders.Builder`.

        **Return**

        * `list( tf.Tensor )`

        ** Example **

        This examples creates a network to that solves the XOR problem using sigmoid units

            import tensorflow as tf
            from tensorbuilder import tb

            x = tf.placeholder(tf.float32, shape=[None, 2])
            y = tf.placeholder(tf.float32, shape=[None, 1])


            #Network
            [activation_tensor, trainer_tensor] = (
                tb.build(x)

                .sigmoid_layer(2)
                .linear_layer(1)

                .branch(lambda logit:
                [
                    logit.sigmoid() # activation
                ,
                    logit
                    .sigmoid_cross_entropy_with_logits(y) # loss
                    .map(tf.train.AdamOptimizer(0.01).minimize) # trainer
                ])
                .tensors()
            )

        Same example using the DSL

            import tensorflow as tf
            from tensorbuilder import tb

            x = tf.placeholder(tf.float32, shape=[None, 2])
            y = tf.placeholder(tf.float32, shape=[None, 1])


            #Network
            [activation_tensor, trainer_tensor] = tb.pipe(
                x,
                tb.sigmoid_layer(2)
                .linear_layer(1),
                [
                    tb.sigmoid() # activation
                ,
                    tb
                    .sigmoid_cross_entropy_with_logits(y) # loss
                    .map(tf.train.AdamOptimizer(0.01).minimize) # trainer
                ],
                tb.tensors()
            )

        """
        return [ builder._tensor for builder in self ]


    @immutable
    def __iter__(tree):
        """A generator function that lazily returns all the Builders contianed by this tree"""
        for branch in tree._branches:
            for builder in branch:
                yield builder



## Module Funs
def _tree_register_reduce_method_docs(original_name, library_path, name, fn_signature, fn_docs):
    return """
THIS METHOD IS AUTOMATICALLY GENERATED

**@immutable**

This method reduces the whole BuilderTree to a single Builder by applying `tensorbuilder.core.builders.BuilderTree.reduce` with `{1}.{0}`.

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

This method is a lifted version the function `{1}.{0}` to work with `tensorbuilder.core.builders.Builder`s. Instead of taking a Tensor as its first argument it takes a builder, the rest of the arguments are exactly the same.


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


## SETUP
BuilderBase.__core__ = [ name for name, f in inspect.getmembers(BuilderBase, predicate=inspect.ismethod) ]
BuilderTreeBase.__core__ = [ name for name, f in inspect.getmembers(BuilderTreeBase, predicate=inspect.ismethod) ]
