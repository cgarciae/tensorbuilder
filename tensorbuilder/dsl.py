"""

"""

import inspect
import utils
import functools
import itertools
import tensorflow as tf
import tensorbuilder as tb

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

    @tb.immutable
    def tensor(app):
        """
        """
        return app.compose(tensor())


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
    if type(ast) is tuple:
        return _sequence_function(ast)
    elif type(ast) is list:
        return _branch_function(ast)
    elif hasattr(ast, '__call__'):
        return ast
    else:
        raise Exception("Element has to be either a tuple for sequential operations, a list for branching, or a function from a builder to a builder")

def pipe(builder, *ast):
    f = compile(ast)
    return f(builder)

def tensor():
    """
    Takes a Builder and returns its tensor
    """
    return lambda builder: builder.tensor


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
    exec("""


def {0}(*args, **kwargs):
	\"\"\"
THIS FUNCTION IS AUTOMATICALLY GENERATED

This function accepts the same arguments as `{3}.{0}` but instead of getting the class instance as its first arguments, it returns a function that expects a builder and applies the builder plus all \*args and \*\*kwargs to `{3}.{0}`. The returned function is an `tensorbuilder.dsl.Applicative`, so you can use all the methods defined by this class.

** utils of `{3}.{0}`**

	def {1}


	\"\"\"

	return Applicative(lambda builder: builder.{0}(*args, **kwargs))


@tb.immutable
def __{0}(app, *args, **kwargs):
    \"\"\"
THIS METHOD IS AUTOMATICALLY GENERATED

This method accepts the same arguments as `{3}.{0}` but:

1. Forwards all of its arguments to `tensorbuilder.dsl.{0}`, this returns a function `g`.
2. Applies `tensorbuilder.dsl.Applicative.compose` over `g`, this roughly computes the composition `g` of `tensorbuilder.dsl.Applicative.f`.

So the result of this method is compose `tensorbuilder.dsl.{0}` with `tensorbuilder.dsl.Applicative.f`.

** utils of `{3}.{0}`**

	def {1}

	\"\"\"
    g = {0}(*args, **kwargs)
    return app.compose(g)

Applicative.{0} = __{0}

 	""".format(_name, _f_signature, _f_docs, _module_name))


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
