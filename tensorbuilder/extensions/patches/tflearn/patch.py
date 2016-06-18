import tflearn
import tensorbuilder as tb
import inspect

for _nn_name, f in inspect.getmembers(tflearn, inspect.isfunction):
    tb.Builder.register_map_method(f, "tflearn")
