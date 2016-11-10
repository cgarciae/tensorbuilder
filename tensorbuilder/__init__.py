# Patch docs
import os
import sys
from builder import Builder
from tensordata import Data
import patches


class TensorBuilder(Builder):
    """docstring for TensorBuilder."""
    def __init__(self):
        super(TensorBuilder, self).__init__()

    def data(self, *args, **kwargs):
        return Data(*args, **kwargs)


tensorbuilder = TensorBuilder()
patches.patch(TensorBuilder)


#pdoc
__all__ = ["tensordata", "patches", "builder"]

#set documentation
def _read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

module = sys.modules[__name__]
raw_docs = _read("README-template.md")
__version__ = _read("version.txt")
module.__doc__ = raw_docs.format(__version__)
