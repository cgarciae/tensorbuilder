from tensorbuilder import TensorBuilder as TB
import tensorflow as tf
import inspect

# register functions
register_1_functions = [
    "matmul"
]

functions_1 = [ (name, f) for name, f in inspect.getmembers(tf, inspect.isfunction) if name in register_1_functions ]

for name, f in functions_1:
    TB.register_function(f, "tf")
