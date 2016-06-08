# Init code
from tensorbuilder import *
import tensorbuilder
import tensorflow as tf

# Monkey Patch TensorFlow's Tensor with a `build` method as `builder`
tf.python.framework.ops.Tensor.builder = build

#version
__version__ = "0.0.1"
__all__ = ["tensorbuilder", "tensorbuilder.patches", "tensorbuilder.patches.tensorflow"]
