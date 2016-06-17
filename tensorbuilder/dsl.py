"""

"""

import inspect
import utils
import functools
import itertools
import tensorflow as tf
import tensorbuilder as tb
import sys

_self = sys.modules[__name__]

#######################
### FUNCTOR
#######################
class Applicative(object):
    """docstring for Applicative"""
    def __init__(self, f):
        super(Applicative, self).__init__()
        self.f = f
        """
        A function of type `a -> b`.
        """

    def copy(self):
        """Returns a compy of the applicative"""
        return Applicative(self.f)

    def __call__(self, x):
        return self.f(x)

    @tb.immutable
    def compose(app, g):

        return Applicative(_compose2(g, app.f))


def applicative(f):
    return Applicative(f)
#######################
### FUNCTIONS
#######################

def _compose2(f, g):
    return lambda x: f(g(x))

_identity = lambda x: x

def _compose_reversed(functions):
    functions = functions[:]
    functions.reverse()
    return functools.reduce(_compose2, functions, _identity)


def _sequence_function(tuple_ast):
    fs = [ compile(ast) for ast in tuple_ast ]
    return _compose_reversed(fs)

def _branch_function(list_ast):
    fs = [ compile(ast) for ast in list_ast ]
    return lambda builder: builder.branch(lambda builder: [ f(builder) for f in fs ])

def compile(ast):
    #if type(ast) is tuple:

    if type(ast) is list:
        return _branch_function(ast)
    elif hasattr(ast, '__call__'):
        return ast
    else:
        return _sequence_function(ast)
        #raise Exception("Element has to be either a tuple for sequential operations, a list for branching, or a function from a builder to a builder, got %s, %s" % (type(ast), type(ast) is tuple))

def pipe(builder, *ast):
    f = compile(ast)
    return f(builder)



def _get_fun(_name, _f_signature, _f_docs, _module_name):
    def _fun(*args, **kwargs):
        def _lambda(builder):
            f = getattr(builder, _name)
            return f(*args, **kwargs)

    	return Applicative(_lambda)

    _fun.__name__ = _name
    _fun.__doc__ = """
    THIS FUNCTION IS AUTOMATICALLY GENERATED

    This function accepts the same arguments as `{3}.{0}` but instead of getting the class instance as its first arguments, it returns a function that expects a builder and applies the builder plus all \*args and \*\*kwargs to `{3}.{0}`. The returned function is an `tensorbuilder.dsl.Applicative`, so you can use all the methods defined by this class.

    ** Documentation for `{3}.{0}`**

        def {1}

    """.format(_name, _f_signature, _f_docs, _module_name)

    return _fun



def _get_method(_name, _f_signature, _f_docs, _module_name):
    @tb.immutable
    def _method(app, *args, **kwargs):
        f = getattr(_self, _name)
        g = f(*args, **kwargs)
        return app.compose(g)

    _method.__name__ = _name
    _method.__doc__ = """
    THIS METHOD IS AUTOMATICALLY GENERATED

    This method accepts the same arguments as `{3}.{0}` but:

    1. Forwards all of its arguments to `tensorbuilder.dsl.{0}`, this returns a function `g`.
    2. Applies `tensorbuilder.dsl.Applicative.compose` over `g`, this roughly computes the composition `g` of `tensorbuilder.dsl.Applicative.f`.

    So the result of this method is compose `tensorbuilder.dsl.{0}` with `tensorbuilder.dsl.Applicative.f`.

    ** utils of `{3}.{0}`**

        def {1}

    """.format(_name, _f_signature, _f_docs, _module_name)

    return _method

#######################
### CODE GENERATION
#######################

_builder_excluded = ["copy"]
def _builder_methods():
    for _name, f in inspect.getmembers(tb.Builder, inspect.ismethod):
        if _name[0] != '_' and _name not in _builder_excluded:
            yield ("Builder", _name, f)

_tree_excluded = ["copy", "connect_layer" ] #, "tensors", "builders"]
def _builder_tree_methods():
    for _name, f in inspect.getmembers(tb.BuilderTree, inspect.ismethod):
        if _name[0] != '_' and _name not in _tree_excluded:
            yield ("BuilderTree", _name, f)


for _module_name, _name, f in itertools.chain(_builder_methods(), _builder_tree_methods()):
    _f_signature = utils.get_method_sig(f)
    _f_docs = inspect.getdoc(f)
    _fun = _get_fun(_name, _f_signature, _f_docs, _module_name)
    _method = _get_method(_name, _f_signature, _f_docs, _module_name)

    setattr(_self, _name, _fun)
    setattr(Applicative, _name, _method)


#######################
### CUSTOM FUNCTIONS
#######################
def identity():
	"""
    Returns the builder unchanged.
	"""
	return Applicative(lambda builder: builder)

#######################
### MONEKY PATCHING
#######################
tb.Builder.pipe = pipe


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
