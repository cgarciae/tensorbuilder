"""

"""

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
    """docstring for Applicative"""

    __metaclass__ = ABCMeta

    def __init__(self, f):
        super(ApplicativeBase, self).__init__()
        self.f = f
        """
        A function of type `a -> b`.
        """

    def Builder(self, tensor):
        pass


    def unit(self, f):
        return self.__class__(f)

    def copy(self):
        """Returns a compy of the applicative"""
        return self.unit(self.f)

    def __call__(self, x):
        return self.f(x)

    def compose(app, g):
        return app.unit(_compose2(g, app.f))


    def identity(self):
    	"""
        Returns the expression unchanged.
    	"""
    	return self

    def pipe(self, builder, *ast):
        f = _compile(ast)

        #if the input is a Tensor, create a Builder
        if type(builder) is tf.python.framework.ops.Tensor:
            builder = self.Builder(builder)

        return f(builder)

    def compile(self, ast):
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

        This method accepts the same arguments as `{3}.{0}

        ** Documentation from `{3}.{0}`**

            def {1}

        """.format(name, fn_signature, fn.__doc__, library_path)


        setattr(cls, name, fn)



ApplicativeBase.__core__ = [ _name for _name, f in inspect.getmembers(ApplicativeBase, predicate=inspect.ismethod) ]

#######################
### FUNCTIONS
#######################

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

    	return app.unit(_lambda)

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

    h = x.builder().pipe(
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
