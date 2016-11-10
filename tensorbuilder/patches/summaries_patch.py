import tensorflow as tf
import inspect
import functools

def patch(TensorBuilder):
    funs = ( (name, f) for (name, f) in inspect.getmembers(tf, inspect.isfunction) if "_summary" in name )

    for name, f in funs:
        @TensorBuilder.register2("tf", alias="make_{0}".format(name), wrapped=f, original_name=name)
        def summary_function(tags, values, *args, **kwargs):
            f(tags, values, *args, **kwargs)
            return values

        #TensorBuilder.register_function2(summary_function, )
