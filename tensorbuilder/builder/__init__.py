import inspect
import utils
import functools
import itertools
import tensorflow as tf
import sys
from types import MethodType
from abc import ABCMeta, abstractmethod

def _identity(x):
    return x

#######################
### Applicative
#######################
class Ref(object):
    """docstring for Ref."""
    def __init__(self, ref=None):
        super(Ref, self).__init__()
        self.ref = ref

    def __call__(self, *optional):
        return self.ref

    def set(self, x):
        self.ref = x
        return x

class Scope(object):
    """docstring for Scope."""
    def __init__(self, new_scope):
        super(Scope, self).__init__()
        self.new_scope = new_scope
        self.old_scope = Builder._S

    def __enter__(self):
        Builder._S = self.new_scope

    def __exit__(self, *args):
        Builder._S = self.old_scope


class Builder(object):
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

    def __init__(self, f=_identity):
        super(Builder, self).__init__()
        ast = (f,)
        self.f = self.compile(*ast)


    def _unit(self, f, _return_type=None):
        "Monadic unit, also known as `return`"
        if _return_type:
            return _return_type(f)
        else:
            return self.__class__(f)

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    _S = None

    @property
    def S(self):
        return Builder._S

    def _(self, g, *args, **kwargs):
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
        _return_type = None

        if '_return_type' in kwargs:
            _return_type = kwargs['_return_type']
            del kwargs['_return_type']

        g = self.compile(g)
        return self._unit(lambda x: g(self.f(x), *args, **kwargs), _return_type=_return_type)

    def _2(self, g, arg1, *args, **kwargs):
        """
        """
        _return_type = None

        if '_return_type' in kwargs:
            _return_type = kwargs['_return_type']
            del kwargs['_return_type']

        g = self.compile(g)

        def _lambda(x):
            arg2 = self.f(x)
            new_args = tuple([arg1, arg2] + list(args))
            return g(*new_args, **kwargs)

        return self._unit(_lambda, _return_type=_return_type)

    def _3(self, g, arg1, arg2, *args, **kwargs):
        """
        """
        _return_type = None

        if '_return_type' in kwargs:
            _return_type = kwargs['_return_type']
            del kwargs['_return_type']

        g = self.compile(g)

        def _lambda(x):
            arg3 = self.f(x)
            new_args = tuple([arg1, arg2, arg3] + list(args))
            return g(*new_args, **kwargs)

        return self._unit(_lambda, _return_type=_return_type)

    def _4(self, g, arg1, arg2, arg3, *args, **kwargs):
        """
        """
        _return_type = None

        if '_return_type' in kwargs:
            _return_type = kwargs['_return_type']
            del kwargs['_return_type']

        g = self.compile(g)

        def _lambda(x):
            arg4 = self.f(x)
            new_args = tuple([arg1, arg2, arg3, arg4] + list(args))
            return g(*new_args, **kwargs)

        return self._unit(_lambda, _return_type=_return_type)

    def _5(self, g, arg1, arg2, arg3, arg4, *args, **kwargs):
        """
        """
        _return_type = None

        if '_return_type' in kwargs:
            _return_type = kwargs['_return_type']
            del kwargs['_return_type']

        g = self.compile(g)

        def _lambda(x):
            arg5 = self.f(x)
            new_args = tuple([arg1, arg2, arg3, arg4, arg5] + list(args))
            return g(*new_args, **kwargs)

        return self._unit(_lambda, _return_type=_return_type)


    def using(self, x):
        """
        """
        return self._(lambda _: x)

    def run(self):
        return self(None)

    def store(self, ref):
        return self._(ref.set)

    @property
    def ref(self):
        return Ref()

    def identity(self, x):
        return x

    def __rrshift__(self, x):
        if isinstance(x, Builder):
            return x._(self.f)
        else:
            return self.f(x)

    @classmethod
    def pipe(cls, x, *ast):
        """
        `pipe` takes in a `builder` of type `Builder`, `BuilderTree` or `Object` preferably and an object `ast` which must be part of the domain of the DSL, and compiles `ast` to a function of type `Builder -> Builder` and applies it to the input `builder`. All \*args after `builder` are taken as a tuple, therefore, it makes it easier to define an initial tuple `()` element to define a sequential operation.

        **Arguments**

        * `builder`: a `Builder`, `BuilderTree` or `Object` preferably.
        * `*ast`: a sequence of elements of the DSL.

        **Return**

        An object with the result of the computation, probable types: `Object | Builder | BuilderTree | list(Object) |  `

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
                .object()
            )
        """

        f = cls.compile(*ast)

        return f(x)

    @classmethod
    def compile(cls, *ast):
        """
        `compile` an object `ast` which must be part of the domain of the DSL and returns function. It applies the rules of the DSL to create an actual Python function that does what you intend. Normally you will just use pipe, which not only compiles the DSL it actually performs the computation to a given Object/Builder, however, it you are building and API this might be useful since you can create a function from an AST which can itself be used as an element of another AST since final elements of the DSL are functions.

        **Arguments**

        * `*ast`: a sequence of elements of the DSL.

        **Return**

        A function

        **Examples**

            import tensorflow as tf
            from tensorbuilder import tb

            x = placeholder(tf.float32, shape=[None, 10])

            f = tb.compile(
                tb.build, #accept a Object as a parameter and create a builder so you can use the rest of the methods
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
                .object()
            )

            h = f(x)

        """

        f = _compile(ast)

        if _is_iterable_ast(f):
            f = _compile_iterable(f)

        return f


    @classmethod
    def register_as_method(cls, fn, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation=""):
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
        if wrapped:
            fn = functools.wraps(wrapped)(fn)

        fn_signature = utils.get_method_sig(fn)
     	fn_docs = inspect.getdoc(fn)
        name = alias if alias else fn.__name__
        original_name = fn.__name__ if wrapped else original_name if original_name else name

        fn.__name__ = name
        fn.__doc__ = doc if doc else ("""
        THIS METHOD IS AUTOMATICALLY GENERATED

            tb.{1}(*args, **kwargs)

        This method accepts the same arguments as `{3}.{0}`. """ + explanation + """

        ** Documentation from `{3}.{0}`**

            {2}
        """).format(original_name, name, fn_docs, library_path)


        setattr(cls, name, fn)

    @classmethod
    def register_method(self, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation=""):
        def register_decorator(fn):

            self.register_as_method(fn, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation)

            return fn
        return register_decorator

    @classmethod
    def register_function(cls, fn, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        """
        This method enables you to register any function `fn` that takes an object as its first argument as a method of the Builder and Applicative class.

        **Arguments**

        * `fn`: a function that atleast takes an Object as its first argument.
        * `library_path`: the route of the librar from which this function was taken, used for documentation purposes.
        * `alias`: allows you to specify the name of the method, it will take the name of the function if its `None`.
        * `doc`: the documentation for the method, if `None` a predefied documentation will be generated based on the documentation of `fn`.

        **Return**

        `None`

        **Examples**

        """
        @functools.wraps(fn)
        def method(self, *args, **kwargs):
            kwargs['_return_type'] = _return_type
            return self._(fn, *args, **kwargs)

        explanation = """However, the 1st argument is omitted, a partial with the rest of the arguments is returned which expects the 1st argument such that

            {3}.{0}(x1, *args, **kwargs) <==> tb.{1}(*args, **kwargs)(x1)

        """ + explanation

        cls.register_as_method(method, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation)

    @classmethod
    def register_function2(cls, fn, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        """
        """
        @functools.wraps(fn)
        def method(self, *args, **kwargs):
            kwargs['_return_type'] = _return_type
            return self._2(fn, *args, **kwargs)

        explanation = """However, the 2nd argument is omitted, a partial with the rest of the arguments is returned which expects the 2nd argument such that

            {3}.{0}(x1, x2, *args, **kwargs) <==> tb.{1}(x1, *args, **kwargs)(x2)
        """ + explanation

        cls.register_as_method(method, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation)

    @classmethod
    def register_function3(cls, fn, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        """
        """
        @functools.wraps(fn)
        def method(self, *args, **kwargs):
            kwargs['_return_type'] = _return_type
            return self._3(fn, *args, **kwargs)

        explanation = """However, the 3rd argument is omitted, a partial with the rest of the arguments is returned which expects the 3rd argument such that

            {3}.{0}(x1, x2, x3, *args, **kwargs) <==> tb.{1}(x1, x2, *args, **kwargs)(x3)
        """ + explanation

        cls.register_as_method(method, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation)

    @classmethod
    def register_function4(cls, fn, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        """
        """
        @functools.wraps(fn)
        def method(self, *args, **kwargs):
            kwargs['_return_type'] = _return_type
            return self._4(fn, *args, **kwargs)

        explanation = """However, the 4th argument is omitted, a partial with the rest of the arguments is returned which expects the 4th argument such that

            {3}.{0}(x1, x2, x3, x4, *args, **kwargs) <==> tb.{1}(x1, x2, x3, *args, **kwargs)(x4)
        """ + explanation

        cls.register_as_method(method, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation)

    @classmethod
    def register_function5(cls, fn, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        """
        """
        @functools.wraps(fn)
        def method(self, *args, **kwargs):
            kwargs['_return_type'] = _return_type
            return self._5(fn, *args, **kwargs)

        explanation = """However, the 5th argument is omitted, a partial with the rest of the arguments is returned which expects the 5th argument such that

            {3}.{0}(x1, x2, x3, x4, x5, *args, **kwargs) <==> tb.{1}(x1, x2, x3, x4, *args, **kwargs)(x5)
        """ + explanation

        cls.register_as_method(method, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation, _return_type=_return_type)

    @classmethod
    def register(cls, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        def register_decorator(fn):
            cls.register_function(fn, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation, _return_type=_return_type)
            return fn
        return register_decorator

    @classmethod
    def register2(cls, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        def register_decorator(fn):
            cls.register_function2(fn, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation, _return_type=_return_type)
            return fn
        return register_decorator

    @classmethod
    def register3(cls, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        def register_decorator(fn):
            cls.register_function3(fn, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation, _return_type=_return_type)
            return fn
        return register_decorator

    @classmethod
    def register4(cls, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        def register_decorator(fn):
            cls.register_function4(fn, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation, _return_type=_return_type)
            return fn
        return register_decorator

    @classmethod
    def register5(cls, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", _return_type=None):
        def register_decorator(fn):
            cls.register_function5(fn, library_path, alias=alias, original_name=original_name, doc=doc, wrapped=wrapped, explanation=explanation, _return_type=_return_type)
            return fn
        return register_decorator


def __(*args, **kwargs):
    return Builder.pipe(*args, **kwargs)

def C(*args, **kwargs):
    return Builder(args)

#######################
### FUNCTIONS
#######################

def _compile(ast):
    #if type(ast) is tuple:
    try:
        if hasattr(ast, '__call__'):
            return ast
        elif type(ast) is tuple:
            return _compile_tuple(ast)
        elif type(ast) is dict:
            return _compile_dictionary(ast)
        else:
            return _compile_iterable(ast) #its iterable
            #raise Exception("Element has to be either a tuple for sequential operations, a list for branching, or a function from a builder to a builder, got %s, %s" % (type(ast), type(ast) is tuple))

    except:
        print(ast)
        raise

def _compose2(ast_f, ast_g):
    g = _compile(ast_g)
    if _is_iterable_ast(ast_f):
        return [ _compose2(_compile(f), g) for f in ast_f ]
    else:
        f = _compile(ast_f)
        return lambda x: f(g(x))

def _is_iterable_ast(ast):
    return hasattr(ast, '__iter__') and not( type(ast) is tuple or type(ast) is dict or hasattr(ast, '__call__') )

def _compile_tuple(tuple_ast):
    if len(tuple_ast) == 1:
        return _compile(tuple_ast[0])

    tuple_ast = list(tuple_ast)
    tuple_ast.reverse()
    tuple_ast = tuple_ast + [ _identity ]

    f = functools.reduce(_compose2, tuple_ast)

    if _is_iterable_ast(f):
        f = utils.flatten_list(f)

    return f

def _compile_iterable(list_ast):
    # try:
    #     list_ast = list(list_ast)
    # except:
    #     import traceback
    #     print(list_ast)
    #     traceback.print_stack()
    #     raise
    list_ast = list(list_ast)
    list_ast = utils.flatten_list(list_ast)
    fs = utils.flatten_list([ _compile(ast) for ast in list_ast ])
    return lambda x: [ f(x) for f in fs ]

def _compile_dictionary(dict_ast):
    scope, body_ast = list(dict_ast.items())[0]
    body = _compile(body_ast)
    def _lambda(x):
        with scope as new_scope:
            with Scope(new_scope):
                return body(x)
    return _lambda



#######################
### CUSTOM FUNCTIONS
#######################


if __name__ == "__main__":
    pass
