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
import core
import tensordata
import extensions
import api

from api import *


#pdoc
__all__ = ["core", "tensordata", "extensions", "api"]
