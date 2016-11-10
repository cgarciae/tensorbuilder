import tensorflow as tf
import inspect

def patch(TensorBuilder):
    # register functions
    register_1_functions = [
        "matmul"
    ]

    functions_1 = [ (name, f) for name, f in inspect.getmembers(tf, inspect.isfunction) if name in register_1_functions ]

    for name, f in functions_1:
        TensorBuilder.register_function(f, "tf")
