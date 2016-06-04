# Init code
import tensorflow as tf
from tensorbuilder import *
import nn


# Monkey Patch TensorFlow
tf.python.framework.ops.Tensor.builder = builder


import tensorbuilder
__all__ = ["tensorbuilder", "nn"]
