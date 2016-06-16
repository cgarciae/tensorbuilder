"""
# Tensor Builder

TensorBuilder is light-weight extensible library that enables you to easily create complex deep neural networks through a functional [fluent](https://en.wikipedia.org/wiki/Fluent_interface) [immutable](https://en.wikipedia.org/wiki/Immutable_object) API based on the Builder Pattern. Tensor Builder also comes with a DSL based on [applicatives](http://learnyouahaskell.com/functors-applicative-functors-and-monoids) and function composition that enables you to express more clearly the structure of your network, make changes faster, and reuse code.

### Goals

* Be a light-wrapper around Tensor-based libraries
* Enable users to easily create complex branched topologies while maintaining a fluent API (see [Builder.branch](http://cgarciae.github.io/tensorbuilder/tensorbuilder.m.html#tensorbuilder.tensorbuilder.Builder.branch))
* Let users be expressive and productive through a DSL

## Installation
Tensor Builder assumes you have a working `tensorflow` installation. We don't include it in the `requirements.txt` since the installation of tensorflow varies depending on your setup.

#### From github
1. `pip install git+https://github.com/cgarciae/tensorbuilder.git@0.0.3`

#### From pip
Coming soon!

## Getting Started

Create neural network with a [5, 10, 3] architecture with a `softmax` output layer and a `tanh` hidden layer through a Builder and then get back its tensor:

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.patch

    x = tf.placeholder(tf.float32, shape=[None, 5])
    keep_prob = tf.placeholder(tf.float32)

    h = (
      x.builder()
      .tanh_layer(10) # tanh(x * w + b)
      .dropout(keep_prob) # dropout(x, keep_prob)
      .softmax_layer(3) # softmax(x * w + b)
      .tensor()
    )

    print(h)

## Features
* **Branches**: Enable to easily express complex complex topologies with a fluent API. See [Branches](http://cgarciae.github.io/tensorbuilder/guide/branches.m.html).
* **Patches**: Add functions from other Tensor-based libraries as methods of the Builder class. TensorBuilder gives you a curated patch plus some specific patches from `TensorFlow` and `TFLearn`, but you can build you own to make TensorBuilder what you want it to be. See [Patches](http://cgarciae.github.io/tensorbuilder/guide/patches.m.html).
* **DSL**: Use an abbreviated notation with a functional style to make the creation of networks faster, structural changes easier, and reuse code. See [DSL](http://cgarciae.github.io/tensorbuilder/guide/dsl.m.html).

## The Guide
Check out the guide [here](http://cgarciae.github.io/tensorbuilder/guide/index.html).

## Documentation

Check out the complete documentation [here](http://cgarciae.github.io/tensorbuilder/).
"""

# Init code
from tensorbuilder import *
import tensorbuilder
import tensorflow as tf
from tensordata import data
import guide

# Uncomment to generate docs only
#import patch
#import dsl

# Monkey Patch TensorFlow's Tensor with a `build` method as `builder`
tf.python.framework.ops.Tensor.builder = build

#version
__version__ = "0.0.2"
__all__ = ["dsl", "tensorbuilder", "batcher", "guide"]
