from builder_class import Builder, utils, C, P, _1, _2
import inspect

Builder.M = property(lambda self: MethodBuilder(self.f))

class MethodBuilder(Builder):
    """docstring for MethodBuilder."""

    def proxy(self, name, *args, **kwargs):
        if not hasattr(self, name):
            register_proxy_method(name)
        return self._unit(lambda x: getattr(x, name)(*args, **kwargs))

M = MethodBuilder()

def register_proxy_method(method_name, alias=None):
    def proxy_method(self, *args, **kwargs):
        return getattr(self, method_name)(*args, **kwargs)

    alias = alias if alias else method_name
    MethodBuilder.register_function_1(proxy_method, "any", alias=alias)

_is_viable_method = lambda m: inspect.isroutine(m) and m.__name__[0] is not '_'
_get_members = _1(inspect.getmembers, _is_viable_method)

method_info_list = P(
    [str, int, list, tuple, dict, float, bool],
    _2(map, _get_members),
    utils.flatten_list
)

for name, f in method_info_list:
    register_proxy_method(name)

# built in functions

function_2_names = ["map", "filter", "reduce"]
functions_2 = [ (name, f) for name, f in __builtins__.items() if name in function_2_names ]

for name, f in __builtins__.items():
    try:
        if hasattr(f, "__name__") and name[0] is not "_" and name not in function_2_names:
            MethodBuilder.register_function_1(f, "python", alias=name)
    except Exception as e:
        print(e)

for name, f in functions_2:
    MethodBuilder.register_function_2(f, "python")


#special methods:
special_methods = {'__contains__': 'contains' }

for method_name, alias in special_methods.items():
    register_proxy_method(method_name, alias)

#custom methods
@MethodBuilder.register_1("python")
def Not(x): return not x

@MethodBuilder.register_1("python")
def And(x, y): return x and y

@MethodBuilder.register_1("python")
def Or(x, y): return x or y
