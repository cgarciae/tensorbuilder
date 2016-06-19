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
1. `pip install git+https://github.com/cgarciae/tensorbuilder.git@0.0.4`

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

## The Guide
Check out the guide [here](http://cgarciae.github.io/tensorbuilder/guide/index.html).

## Documentation
* Complete documentation: [here](http://cgarciae.github.io/tensorbuilder/).
* Builder API: [here](http://cgarciae.github.io/tensorbuilder/tensorbuilder.m.html).

## Features
* **Branches**: Enable to easily express complex complex topologies with a fluent API. See [Branches](http://cgarciae.github.io/tensorbuilder/guide/branches.m.html).
* **Patches**: Add functions from other Tensor-based libraries as methods of the Builder class. TensorBuilder gives you a curated patch plus some specific patches from `TensorFlow` and `TFLearn`, but you can build you own to make TensorBuilder what you want it to be. See [Patches](http://cgarciae.github.io/tensorbuilder/guide/patches.m.html).
* **DSL**: Use an abbreviated notation with a functional style to make the creation of networks faster, structural changes easier, and reuse code. See [DSL](http://cgarciae.github.io/tensorbuilder/guide/dsl.m.html).

#### Showoff
This next is some more involved code for solving the MNIST for images of 20x20 gray-scaled. It is solve using a using 3 relu-CNN branches with max pooling, the branches are merged through a fully connected relu layer with dropout, and finally its connected to a softmax output layer.

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.patch

    # Define variables
    y = tf.placeholder(tf.float32, shape=[None, 400])
    x = tf.placeholder(tf.float32, shape=[None, 400])
    keep_prob = tf.placeholder(tf.float32)

    #Create the convolution function to be used by each brach
    conv_branch = (

        dl.convolution2d(32, [5, 5])
        .relu()
        .max_pool_2d(2) #This method is taken from `tflearn`

        .convolution2d(64, [5, 5])
        .relu()
        .max_pool_2d(2)

        .reshape([-1, 5 * 5 * 64]) #notice that we flatten at the end
    )

    [h, loss, trainer] = x.builder().pipe(

        dl.reshape([-1, 20, 20, 1]),
        [
            conv_branch #Reuse code
        ,
            conv_branch
        ,
            conv_branch
        ],

        dl.relu_layer(1024) # this fully connects all 3 branches into a single relu layer
        .dropout(keep_prob)

        .fully_connected(10), # create a linear connection
        [
            dl.softmax() # h
        ,
            (dl.softmax_cross_entropy_with_logits(y)
            .map(tf.reduce_mean), #calculte loss
            [
                dl.identity() # loss
            ,
                dl.map(tf.train.AdadeltaOptimizer(0.01).minimize) # trainer
            ])
        ],
        dl.tensors() #get the list of tensors from the previous BuilderTree
    )

    print h, loss, trainer

Notice that:
1. We where able to reuse code by specifying the logic for the branches separately using the same syntax
2. Branches are expressed naturally as a list thanks to the DSL, the indentation levels match the depth of the tree. Nested branches are just as easy.
3. Most methods presented are functions from `tensorflow` that your are (probably) used to.


## Examples

Here are many examples to you give a taste of what it feels like to use TensorBuilder and teach you some basic patterns.
"""

# Init code
import tensorflow as tf
from core import BuilderBase, BuilderTreeBase, ApplicativeBase
import tensordata
import guide
import extensions

Builder, BuilderTree, Applicative = extensions.tensorbuilder_classes()


__builder__ = Builder(None)
__tree__ = BuilderTree([])
__applicative__ = Applicative(lambda x: x)

build = __builder__.build
branches = __tree__.branches
pipe = __applicative__.pipe
data = tensordata.data
dl = __applicative__

# Monkey Patch TensorFlow's Tensor with a `build` method as `builder`
# tf.python.framework.ops.Tensor.builder = build

#version
__version__ = "0.0.5"
__all__ = ["core", "guide", "tensordata", "extensions"]
