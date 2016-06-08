# Init code
from tensorbuilder import *
import tensorbuilder
import tensorflow as tf

# uncomment to generate docs only
import patch
import dsl

# Monkey Patch TensorFlow's Tensor with a `build` method as `builder`
tf.python.framework.ops.Tensor.builder = build

#version
__version__ = "0.0.2"
__all__ = ["dsl", "tensorbuilder"]
