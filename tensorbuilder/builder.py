from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from phi import Builder
import inspect
from .tensordata import Data

class TensorBuilder(Builder):
    """docstring for TensorBuilder."""

    def data(self, *args, **kwargs):
        return Data(*args, **kwargs)

TensorBuilder.__core__ = [ name for name, f in inspect.getmembers(TensorBuilder, predicate=inspect.ismethod) ]

T = TensorBuilder()
