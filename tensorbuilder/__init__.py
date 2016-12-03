from builder import TensorBuilder
from tensordata import Data
from phi import utils

import patches #import last

tensorbuilder = TensorBuilder(utils.identity, {})

########################
# Documentation
########################
import os
import sys

#pdoc
__all__ = ["tensordata", "patches"]

#set documentation
def _read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

module = sys.modules[__name__]
raw_docs = _read("README-template.md")
__version__ = _read("version.txt").split("\n")[0]
module.__doc__ = raw_docs.format(__version__)
