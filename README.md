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

##############################
##### FUNCTIONS
##############################


##############################
##### builder
##############################

The following example shows you how to construct a `tensorbuilder.tensorbuilder.Builder` from a tensorflow Tensor.

    import tensorflow as tf
    import tensorbuilder as tb

    a = tf.placeholder(tf.float32, shape=[None, 8])
    a_builder = tb.build(a)

    print(a_builder)

The previous is the same as

    a = tf.placeholder(tf.float32, shape=[None, 8])
    a_builder = a.builder()

    print(a_builder)

##############################
##### branches
##############################

Given a list of Builders and/or BuilderTrees you construct a `tensorbuilder.tensorbuilder.BuilderTree`.

    import tensorflow as tf
    import tensorbuilder as tb

    a = tf.placeholder(tf.float32, shape=[None, 8]).builder()
    b = tf.placeholder(tf.float32, shape=[None, 8]).builder()

    tree = tb.branches([a, b])

    print(tree)

`tensorbuilder.tensorbuilder.BuilderTree`s are usually constructed using `tensorbuilder.tensorbuilder.Builder.branch` of the `tensorbuilder.tensorbuilder.Builder` class, but you can use this for special cases



##############################
##### BUILDER
##############################


##############################
##### fully_connected
##############################

This method is included by many libraries so its "sort of" part of TensorBuilder. The following builds the computation `tf.nn.sigmoid(tf.matmul(x, w) + b)`

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 5])

    h = (
    	x.builder()
    	.fully_connected(3, activation_fn=tf.nn.sigmoid)
    	.tensor()
    )

    print(h)

Using `tensorbuilder.patch` the previous is equivalent to

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.patch

    x = tf.placeholder(tf.float32, shape=[None, 5])

    h = (
    	x.builder()
    	.sigmoid_layer(3)
    	.tensor()
    )

    print(h)


You can chain various `fully_connected`s to get deeper neural networks

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 40])

    h = (
    	x.builder()
    	.fully_connected(100, activation_fn=tf.nn.tanh)
    	.fully_connected(30, activation_fn=tf.nn.softmax)
    	.tensor()
    )

    print(h)

Using `tensorbuilder.patch` the previous is equivalent to

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.patch

    x = tf.placeholder(tf.float32, shape=[None, 5])

    h = (
    	x.builder()
    	.tanh_layer(100)
    	.softmax_layer(30)
    	.tensor()
    )

    print(h)

##############################
##### map
##############################

The following constructs a neural network with the architecture `[40 input, 100 tanh, 30 softmax]` and and applies `dropout` to the tanh layer

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 40])
    keep_prob = tf.placeholder(tf.float32)

    h = (
    	x.builder()
    	.fully_connected(100, activation_fn=tf.nn.tanh)
    	.map(tf.nn.dropout, keep_prob)
    	.fully_connected(30, activation_fn=tf.nn.softmax)
    	.tensor()
    )

    print(h)


##############################
##### then
##############################

The following *manually* constructs the computation `tf.nn.sigmoid(tf.matmul(x, w) + b)` while updating the `tensorbuilder.tensorbuiler.Builder.variables` dictionary.

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 40])
    keep_prob = tf.placeholder(tf.float32)

    def sigmoid_layer(builder, size):
    	x = builder.tensor()
    	m = int(x.get_shape()[1])
    	n = size

    	w = tf.Variable(tf.random_uniform([m, n], -1.0, 1.0))
    	b = tf.Variable(tf.random_uniform([n], -1.0, 1.0))

    	y = tf.nn.sigmoid(tf.matmul(x, w) + b)

    	return y.builder()

    h = (
    	x.builder()
    	.then(sigmoid_layer, 3)
    	.tensor()
    )

Note that the previous if equivalent to

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch
    h = (
    	x.builder()
    	.fully_connected(3, activation_fn=tf.nn.sigmoid)
    	.tensor()
    )

    print(h)

##############################
##### branch
##############################

The following will create a sigmoid layer but will branch the computation at the logit (z) so you get both the output tensor `h` and `trainer` tensor. Observe that first the logit `z` is calculated by creating a linear layer with `fully_connected(1)` and then its branched out

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 5])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    [h, trainer] = (
        x.builder()
        .fully_connected(1)
        .branch(lambda z:
        [
            z.map(tf.nn.sigmoid)
        ,
            z.map(tf.nn.sigmoid_cross_entropy_with_logits, y)
            .map(tf.train.AdamOptimizer(0.01).minimize)
        ])
        .tensors()
    )

    print(h)
    print(trainer)

Note that you have to use the `tensorbuilder.tensorbuilder.BuilderTree.tensors` method from the `tensorbuilder.tensorbuilder.BuilderTree` class to get the tensors back.

Remember that you can also contain `tensorbuilder.tensorbuilder.BuilderTree` elements when you branch out, this means that you can keep branching inside branch. Don't worry that the tree keep getting deeper, `tensorbuilder.tensorbuilder.BuilderTree` has methods that help you flatten or reduce the tree.
The following example will show you how create a (overly) complex tree and then connect all the leaf nodes to a single `sigmoid` layer

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 5])
    keep_prob = tf.placeholder(tf.float32)

    h = (
        x.builder()
        .fully_connected(10)
        .branch(lambda base:
        [
            base
            .fully_connected(3, activation_fn=tf.nn.relu)
        ,
            base
            .fully_connected(9, activation_fn=tf.nn.tanh)
            .branch(lambda base2:
            [
            	base2
            	.fully_connected(6, activation_fn=tf.nn.sigmoid)
            ,
            	base2
            	.map(tf.nn.dropout, keep_prob)
            	.fully_connected(8, tf.nn.softmax)
            ])
        ])
        .fully_connected(6, activation_fn=tf.nn.sigmoid)
    )

    print(h)

##############################
##### BUILDER TREE
##############################

##############################
##### builders
##############################

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 5])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    [h_builder, trainer_builder] = (
        x.builder()
        .fully_connected(1)
        .branch(lambda z:
        [
            z.map(tf.nn.sigmoid)
        ,
            z.map(tf.nn.sigmoid_cross_entropy_with_logits, y)
            .map(tf.train.AdamOptimizer(0.01).minimize)
        ])
        .builders()
    )

    print(h_builder)
    print(trainer_builder)

##############################
##### tensors
##############################

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 5])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    [h_tensor, trainer_tensor] = (
        x.builder()
        .fully_connected(1)
        .branch(lambda z:
        [
            z.map(tf.nn.sigmoid)
        ,
            z.map(tf.nn.sigmoid_cross_entropy_with_logits, y)
            .map(tf.train.AdamOptimizer(0.01).minimize)
        ])
        .tensors()
    )

    print(h_tensor)
    print(trainer_tensor)

##############################
##### fully_connected
##############################

The following example shows you how to connect two tensors (rather builders) of different shapes to a single `softmax` layer of shape [None, 3]

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    a = tf.placeholder(tf.float32, shape=[None, 8]).builder()
    b = tf.placeholder(tf.float32, shape=[None, 5]).builder()

    h = (
    	tb.branches([a, b])
    	.fully_connected(3, activation_fn=tf.nn.softmax)
    )

    print(h)

The next example show you how you can use this to pass the input layer directly through one branch, and "analyze" it with a `tanh layer` filter through the other, both of these are connect to a single `softmax` output layer

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 5])

    h = (
    	x.builder()
    	.branch(lambda x:
    	[
    		x
    	,
    		x.fully_connected(10, activation_fn=tf.nn.tanh)
    	])
    	.fully_connected(3, activation_fn=tf.nn.softmax)
    )

    print(h)
