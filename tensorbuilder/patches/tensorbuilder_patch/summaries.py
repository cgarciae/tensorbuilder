import inspect
import tensorflow as tf
from tensorbuilder.core import utils

builders_blacklist = [ name for name, f in inspect.getmembers(tf, inspect.isfunction) if "summary" in name ]

def _get_summery_method(f):
    def summary_method(builder, tag, *args, **kwargs):
        f(tag, builder.tensor(), *args, **kwargs)
        return builder

    return summary_method

def patch_classes(Builder, BuilderTree, Applicative):

    summery_methods = [ (name, f) for name, f in inspect.getmembers(tf, inspect.isfunction) if "summary" in name ]

    for name, f in summery_methods:
        f_signature = utils.get_method_sig(f)
        f_docs = inspect.getdoc(f)

        method = _get_summery_method(f)

        method.__name__ = name
        method.__doc__ = """
THIS METHOD IS AUTOMATICALLY GENERATED

Same as `tf.{1}` but the the with the summery tensor as its first parameter.

**Return**

Builder

**Origial documentation for tf.{0}**

    def {1}:

{2}
        """.format(name, f_signature, f_docs)

        # Builder
        Builder.register_method(method, name, doc=method.__doc__) #This should go first