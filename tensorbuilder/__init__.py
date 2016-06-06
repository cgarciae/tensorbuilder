# Init code
import tensorflow as tf
import builders
from builders import *
import nn
import builder_nn
from decorator import decorator

#version
__version__ = "0.0.1"


# Monkey Patch TensorFlow
tf.python.framework.ops.Tensor.builder = builder

__all__ = ["builders", "nn", "builder_nn"]
