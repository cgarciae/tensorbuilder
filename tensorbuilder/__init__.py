# Patch docs
import os
import sys

def _read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

module = sys.modules[__name__]
raw_docs = _read("README-template.md")
__version__ = _read("version.txt")
module.__doc__ = raw_docs.format(__version__)


# Init code


print("aca")

from builder_class import Builder
from tensorbuilder_class import TensorBuilder

tensorbuilder = TensorBuilder()

import patches #do this at the end

#pdoc
__all__ = ["tensordata", "patches"]
