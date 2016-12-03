import tensorflow as tf
import inspect
import functools
from tensorbuilder import TensorBuilder


funs = ( (name, f) for (name, f) in inspect.getmembers(tf, inspect.isfunction) if "_summary" in name )

def register_summary_functions(name, f):
    @TensorBuilder.Register2("tf", alias="make_{0}".format(name), wrapped=f)
    def summary_function(tags, values, *args, **kwargs):
        f(tags, values, *args, **kwargs)
        return values

for name, f in funs:
    register_summary_functions(name, f)
