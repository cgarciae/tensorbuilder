import inspect
import utils
import functools
import itertools
import tensorflow as tf
import sys
from copy import deepcopy, copy
from types import MethodType
from utils import immutable
from abc import ABCMeta, abstractmethod

def _identity(x):
    return x

#######################
### Applicative
#######################
class ApplicativeBase(object):
    """
    An [Applicative](http://learnyouahaskell.com/functors-applicative-functors-and-monoids) is an object who wraps around a function and posses the means to apply it.

    The `Applicative` class contains a inner field `f` that must be a function, internal methods rely on this fact to give you the nice syntax of the DSL. The `Applicative` class is also a function, meaning it implements the `__call__` method, which very simply delegates the computation to the function it contains.

    > **Note:** The `tb` object with is contains the whole TensorBuilder API is an Applicative itself, it contians the identity function.

    **DSL**

    Check out the description of the DSL [here](https://cgarciae.gitbooks.io/tensorbuilder/content/dsl/).

    **Properties**

    Many methods registered/patched by TensorBuilder onto `Applicative` actually use `tensorbuilder.core.applicative.Applicative.compose` internally, therefore, an expression of the DSL like this

        (tb.softmax(),
        tb.dropout(keep_prob),
        tb.relu_layer(10)) # Notice the begging and ending '()' tuple parenthesis

    is equivalent to this

        tb.softmax()
        .dropout(keep_prob),
        .relu_layer(10)


    """

    __metaclass__ = ABCMeta

    def __init__(self, f):
        super(ApplicativeBase, self).__init__()
        self.f = f
        """
        A function of type `a -> b`.
        """

    def Builder(self, tensor):
        pass


    def _unit(self, f):
        "Monadic unit, also known as `return`"
        return self.__class__(f)

    def copy(self):
        """Returns a compy of the applicative"""
        return self._unit(self.f)

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def compose(app, g, *args, **kwargs):
        """
        Takes in a function `g` and composes it with `tensorbuilder.core.Applicative.f` as `g o f`. All \*args and \*\* are forwarded to g. This is an essential method since most registered methods use this.

        **Arguments**

        * `g`: A function
        * All \*args and \*\* are forwarded to `g`

        **Return**

        Applicative

        **Examples**

            import tensorflow as tf
            from tensorbuilder import tb


        """
        return app._unit(lambda x: g(app.f(x), *args, **kwargs))

    def pipe(self, builder, *ast):
        """
        `pipe` takes in a `builder` of type `Builder`, `BuilderTree` or `Tensor` preferably and an object `ast` which must be part of the domain of the DSL, and compiles `ast` to a function of type `Builder -> Builder` and applies it to the input `builder`. All \*args after `builder` are taken as a tuple, therefore, it makes it easier to define an initial tuple `()` element to define a sequential operation.

        **Arguments**

        * `builder`: a `Builder`, `BuilderTree` or `Tensor` preferably.
        * `*ast`: a sequence of elements of the DSL.

        **Return**

        An object with the result of the computation, probable types: `Tensor | Builder | BuilderTree | list(Tensor) |  `

        **Examples**

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            h = tb.pipe(
                x,
                [
                    { tf.device("/gpu:0"):
                        tb.relu_layer(20)
                    }
                ,
                    { tf.device("/gpu:1"):
                        tb.sigmoid_layer(20)
                    }
                ,
                    { tf.device("/cpu:0"):
                        tb.tanh_layer(20)
                    }
                ],
                tb.relu_layer(10)
                .tensor()
            )
        """

        f = _compile(ast)

        #if the input is a Tensor, create a Builder
        if type(builder) is tf.Tensor or type(builder) is tf.Variable:
            builder = self.Builder(builder)

        return f(builder)

    def compile(self, *ast):
        """
        `compile` an object `ast` which must be part of the domain of the DSL and returns function. It applies the rules of the DSL to create an actual Python function that does what you intend. Normally you will just use pipe, which not only compiles the DSL it actually performs the computation to a given Tensor/Builder, however, it you are building and API this might be useful since you can create a function from an AST which can itself be used as an element of another AST since final elements of the DSL are functions.

        **Arguments**

        * `*ast`: a sequence of elements of the DSL.

        **Return**

        A function

        **Examples**

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            f = tb.compile(
                tb.build, #accept a Tensor as a parameter and create a builder so you can use the rest of the methods
                [
                    { tf.device("/gpu:0"):
                        tb.relu_layer(20)
                    }
                ,
                    { tf.device("/gpu:1"):
                        tb.sigmoid_layer(20)
                    }
                ,
                    { tf.device("/cpu:0"):
                        tb.tanh_layer(20)
                    }
                ],
                tb.relu_layer(10)
                .tensor()
            )

            h = f(x)

        """
        return _compile(ast)

    @classmethod
    def register_method(cls, fn, library_path, alias=None, doc=None):
        """
        This method enables you to register any function `fn` that takes an Applicative as its first argument as a method of the Builder class.

        **Arguments**

        * `fn`: a function that atleast takes an Applicative as its first argument.
        * `library_path`: the route of the librar from which this function was taken, used for documentation purposes.
        * `alias`: allows you to specify the name of the method, it will take the name of the function if its `None`.
        * `doc`: the documentation for the method, if `None` a predefied documentation will be generated based on the documentation of `fn`.

        **Return**

        `None`

        **Examples**

        """
        fn_signature = utils.get_method_sig(fn)
     	fn_docs = inspect.getdoc(fn)
        original_name = fn.__name__
        name = alias if alias else original_name

        fn.__name__ = name
        fn.__doc__ = doc if doc else """
        THIS METHOD IS AUTOMATICALLY GENERATED

        This method accepts the same arguments as `{3}.{0}`

        ** Documentation from `{3}.{0}`**

            def {1}
        """.format(name, fn_signature, fn.__doc__, library_path)


        setattr(cls, name, fn)

    @classmethod
    def register_tensor_method(cls, fn, library_path, alias=None, doc=None):
        """
        This method enables you to register any function `fn` that takes an tensor as its first argument as a method of the Builder and Applicative class.

        **Arguments**

        * `fn`: a function that atleast takes an Tensor as its first argument.
        * `library_path`: the route of the librar from which this function was taken, used for documentation purposes.
        * `alias`: allows you to specify the name of the method, it will take the name of the function if its `None`.
        * `doc`: the documentation for the method, if `None` a predefied documentation will be generated based on the documentation of `fn`.

        **Return**

        `None`

        **Examples**

        """
        original_name = fn.__name__
        name = alias if alias else original_name
        method = get_app_method(name)

        cls.register_method(method, library_path, alias=name, doc=doc)

ApplicativeBase.__core__ = [ _name for _name, f in inspect.getmembers(ApplicativeBase, predicate=inspect.ismethod) ]

#######################
### FUNCTIONS
#######################

def get_app_method(name):
    def _method(app, *args, **kwargs):
        def _lambda(builder):
            g = getattr(builder, name)
            return g(*args, **kwargs)
        return app.compose(_lambda)
    return _method

def _compile(ast):
    #if type(ast) is tuple:

    if type(ast) is list:
        return _branch_function(ast)
    elif hasattr(ast, '__call__'):
        return ast
    elif type(ast) is dict:
        return _with_function(ast)
    else:
        return _sequence_function(ast)
        #raise Exception("Element has to be either a tuple for sequential operations, a list for branching, or a function from a builder to a builder, got %s, %s" % (type(ast), type(ast) is tuple))


def _compose2(f, g):
    return lambda x: f(g(x))


def _compose_reversed(functions):
    functions = functions[:]
    functions.reverse()
    return functools.reduce(_compose2, functions, _identity)


def _sequence_function(tuple_ast):
    fs = [ _compile(ast) for ast in tuple_ast ]
    return _compose_reversed(fs)

def _branch_function(list_ast):
    fs = [ _compile(ast) for ast in list_ast ]
    return lambda builder: builder.branch(lambda builder: [ f(builder) for f in fs ])

def _with_function(dict_ast):
    scope, body_ast = list(dict_ast.items())[0]
    body = _compile(body_ast)
    return lambda builder: builder.then_with(lambda: scope)(body)




def _get_fun(_name, _f_signature, _f_docs, _module_name):
    def _fun(app, *args, **kwargs):
        def _lambda(builder):
            f = getattr(builder, _name)
            return f(*args, **kwargs)

    	return app._unit(_lambda)

    _fun.__name__ = _name
    _fun.__doc__ = """
    THIS FUNCTION IS AUTOMATICALLY GENERATED

    This function accepts the same arguments as `{3}.{0}` but instead of getting the class instance as its first arguments, it returns a function that expects a builder and applies the builder plus all \*args and \*\*kwargs to `{3}.{0}`. The returned function is an `tensorbuilder.dsl.Applicative`, so you can use all the methods defined by this class.

    ** Documentation for `{3}.{0}`**

        def {1}

    """.format(_name, _f_signature, _f_docs, _module_name)

    return _fun



#######################
### CUSTOM FUNCTIONS
#######################


if __name__ == "__main__":
    import tensorflow as tf

    x = tf.placeholder(tf.float32, shape=[None, 4])

    h = tb.build(x).pipe(
        connect_layer(10, fn=tf.nn.relu),
        [
            connect_layer(10, fn=tf.nn.relu)
            .connect_layer(5, fn=tf.nn.relu)
        ,
            connect_layer(10, fn=tf.nn.sigmoid)
        ],
        connect_layer(10, fn=tf.nn.relu),
        [
            connect_layer(1)
        ,
            connect_layer(1)
        ],
        tensors
    )

    print(h)
