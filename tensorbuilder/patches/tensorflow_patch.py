import tensorflow as tf
import inspect

def patch(TensorBuilder):
    # register functions

    all_fs = [ name for name, f in (
        inspect.getmembers(tf, inspect.isfunction) +
        inspect.getmembers(tf.nn, inspect.isfunction)
    )]

    f2_names = (
        ["concat"] +
        [ name for name in all_fs if "_summary" in name ]
    )

    f2s = (
        [ (name, f, "tf.nn") for (name, f) in inspect.getmembers(tf.nn, inspect.isfunction) if name in f2_names ] +
        [ (name, f, "tf") for (name, f) in inspect.getmembers(tf, inspect.isfunction) if name in f2_names ]
    )

    f1_blacklist = (
        ["relu_layer", "device"] +
        TensorBuilder.__core__ +
        f2_names
    )

    f1s = (
        [ (name, f, "tf.nn") for (name, f) in inspect.getmembers(tf.nn, inspect.isfunction) if name not in f1_blacklist ] +
        [ (name, f, "tf") for (name, f) in inspect.getmembers(tf, inspect.isfunction) if name not in f1_blacklist ]
    )

    for name, f, module in f1s:
        TensorBuilder.register_function(f, module)

    for name, f, module in f2s:
        TensorBuilder.register_function2(f, module)
