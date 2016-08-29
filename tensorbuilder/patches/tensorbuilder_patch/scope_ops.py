import tensorflow as tf
import inspect
from tensorbuilder.core import utils

builders_blacklist = ["variable_scope", "device"]

def patch_classes(Builder, BuilderTree, Applicative):
    def get_scope_method(f):
        def scope_method(builder, *args, **kwargs):
            return builder.then_with(f, *args, **kwargs)
        return scope_method


    scope_funs = (
        [ (name, f, "tf") for (name, f) in inspect.getmembers(tf, inspect.isfunction) if name in builders_blacklist ]
    )


    for name, f, module_name in scope_funs:
        method_name = "with_" + name
        f_signature = utils.get_method_sig(f)
        f_docs = inspect.getdoc(f)

        scope_method = get_scope_method(f)

        scope_method.__name__ = method_name
        scope_method.__doc__ = """
THIS METHOD IS AUTOMATICALLY GENERATED

Alias for `.then_with({1}.{0}, ...)`

**Arguments**

* All other \*args and \*\*kwargs are forwarded to `tf.contrib.layers.fully_connected`

**Return**

Function of type `(Builder -> Builder) -> Builder`

**Origial documentation for {1}.{0}**

    def {2}:

{3}
        """.format(name, module_name, f_signature, f_docs)

        # Builder
        Builder.register_method(scope_method, module_name, alias=method_name, doc = scope_method.__doc__)