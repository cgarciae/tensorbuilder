from phi.builder import Builder
import inspect
from tensordata import Data
from phi import P

class TensorBuilder(Builder):
    """docstring for TensorBuilder."""

    def data(self, *args, **kwargs):
        return Data(*args, **kwargs)

TensorBuilder.__core__ = [ name for name, f in inspect.getmembers(TensorBuilder, predicate=inspect.ismethod) ]
