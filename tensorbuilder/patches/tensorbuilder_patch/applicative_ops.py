import tensorflow as tf
import inspect
from tensorbuilder.core import utils

builders_blacklist = []

def patch_classes(Builder, BuilderTree, Applicative,
    applicative_builder_blacklist, applicative_tree_blacklist):


    def _get_app_method(f):
        def _method(app, *args, **kwargs):
            def _lambda(builder):
                g = getattr(builder, f.__name__)
                return g(*args, **kwargs)
            return app.compose(_lambda)
        return _method


    _dsl_funs = (
        [ ("BuilderTree", _name, f) for  _name, f in inspect.getmembers(BuilderTree, inspect.ismethod) if _name[0] != '_' and _name not in applicative_tree_blacklist ] +
        [ ("Builder", _name, f) for  _name, f in inspect.getmembers(Builder, inspect.ismethod) if _name[0] != '_' and _name not in applicative_builder_blacklist ]
    )

    for _module_name, _name, f in _dsl_funs:
        _f_signature = utils.get_method_sig(f)
        _f_docs = inspect.getdoc(f)

        _method = _get_app_method(f)

        _method.__name__ = _name
        _method.__doc__ = """
THIS METHOD IS AUTOMATICALLY GENERATED

Alias for `.compose({1}.{0}, ...)`

**Arguments**

* All other \*args and \*\*kwargs are forwarded to `{1}.{0}`

**Return**

Applicative

**Origial documentation for {1}.{0}**

    def {2}:

{3}
        """.format(_name, _module_name, _f_signature, _f_docs)

        Applicative.register_method(_method, _module_name, doc = _method.__doc__)