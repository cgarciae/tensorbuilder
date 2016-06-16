# Tensor Builder

TensorBuilder is light-weight extensible library that enables you to easily create complex deep neural networks through a functional [fluent](https://en.wikipedia.org/wiki/Fluent_interface) [immutable](https://en.wikipedia.org/wiki/Immutable_object) API based on the Builder Pattern. Tensor Builder also comes with a DSL based on [applicatives](http://learnyouahaskell.com/functors-applicative-functors-and-monoids) and function composition that enables you to express more clearly the structure of your network, make changes faster, and reuse code.

Tensor Builder has the following goals:

* Be a light-wrapper around Tensor-based libraries
* Enable users to easily create complex branched topologies while maintaining a fluent API (see [Builder.branch](http://cgarciae.github.io/tensorbuilder/tensorbuilder.m.html#tensorbuilder.tensorbuilder.Builder.branch))
* Let users be expressive and productive through a DSL

http://stackoverflow.com/a/194207/2118130

## Installation
Tensor Builder assumes you have a working `tensorflow` installation. We don't include it in the `requirements.txt` since the installation of tensorflow varies depending on your setup.

#### From github
1. `pip install git+https://github.com/cgarciae/tensorbuilder.git@0.0.2`

#### From pip
Coming soon!

## Getting Started

Create neural network with a [5, 10, 3] architecture with a `softmax` output layer and a `tanh` hidden layer through a Builder and then get back its tensor:

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 5])
    keep_prob = tf.placeholder(tf.float32)

    h = (
    	x.builder()
    	.fully_connected(10, activation_fn=tf.nn.tanh) # tanh(x * w + b)
    	.map(tf.nn.dropout, keep_prob) # dropout(x, keep_prob)
    	.fully_connected(3, activation_fn=tf.nn.softmax) # softmax(x * w + b)
    	.tensor()
    )

    print(h)

Note that `fully_connected` is actually a function from `tf.contrib.layers`, it is patched as a method by the `tensorbuilder.slim_patch`. The `tensorbuilder.patch` includes a lot more methods that register functions from the `tf`, `tf.nn` and `tf.contrib.layers` modules plus some custom methods based on `fully_connected` to create layers:

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

## Patches
Since the Builder class is mostly a [monadic](http://stackoverflow.com/a/194207/2118130) structure which helps you build the computation, in the interest of letting other libraries use Tensor Builder to obtain a fluid API + DSL, TensorBuilder ships the Builder class with no Tensor specific methods but instead contains helpers which enable you to register external functions as methods. Library authors are encouraged to create patches so they can worry about the basic operations and let TensorBuilder add the syntax.

However, TensorBuilder ships with its own main patch, the `tensorbuilder.patch` which adds methods focused on helping you to easily create complex network, it does so (mostly) by registering (cherry picking) methods from other libraries. The intention here is get you the best of what is out there. Here is the list of all the patches you can use.

* `import tensorbuilder.patch`
* `import tensorbuilder.slim_patch`
* `import tensorbuilder.patches.tensorflow.patch`
* `import tensorbuilder.patches.tensorflow.slim`
* `import tensorbuilder.patches.tflearn.patch`

Check out is an example using the `tflearn` patch

    import tflearn
    import tensorbuilder as tb
    import tensorbuilder.patches.tflearn.patch

    model = (
    	tflearn.input_data(shape=[None, 784]).builder()
    	.fully_connected(64)
    	.dropout(0.5)
    	.fully_connected(10, activation='softmax')
    	.regression(optimizer='adam', loss='categorical_crossentropy')
    	.map(tflearn.DNN)
    	.tensor()
    )

    print(model)

## Branching
Branching is common in many neural networks that need to resolve complex tasks because each branch to specialize its knowledge while lowering number of weight compared to a network with wider layers, thus giving better performance. TensorBuilder enables you to easily create nested branches. Branching results in a `tensorbuilder.tensorbuilder.BuilderTree`, which has methods for traversing all the `Builder` leaf nodes and reducing the whole tree to a single `Builder`.

To create a branch you just have to use the `tensorbuilder.tensorbuilder.Builder.branch` method

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.slim_patch

    x = tf.placeholder(tf.float32, shape=[None, 5])
    keep_prob = tf.placeholder(tf.float32)

    h = (
        x.builder()
        .fully_connected(10)
        .branch(lambda root:
        [
            root
            .fully_connected(3, activation_fn=tf.nn.relu)
        ,
            root
            .fully_connected(9, activation_fn=tf.nn.tanh)
            .branch(lambda root2:
            [
              root2
              .fully_connected(6, activation_fn=tf.nn.sigmoid)
            ,
              root2
              .map(tf.nn.dropout, keep_prob)
              .fully_connected(8, tf.nn.softmax)
            ])
        ])
        .fully_connected(6, activation_fn=tf.nn.sigmoid)
        .tensor()
    )

    print(h)

Thanks to TensorBuilder's immutable API, each branch is independent. The previous can also be simplified with the full `patch`

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.patch

    x = tf.placeholder(tf.float32, shape=[None, 5])
    keep_prob = tf.placeholder(tf.float32)

    h = (
        x.builder()
        .fully_connected(10)
        .branch(lambda root:
        [
            root
            .relu_layer(3)
        ,
            root
            .tanh_layer(9)
            .branch(lambda root2:
            [
              root2
              .sigmoid_layer(6)
            ,
              root2
              .dropout(keep_prob)
              .softmax_layer(8)
            ])
        ])
        .sigmoid_layer(6)
        .tensor()
    )

    print(h)

## DSL
TensorBuilder includes a dsl to make creating complex neural networks even easier. Use it if you are experimenting with complex architectures with branches, otherwise stick to basic API. The DSL has these rules:

* All elements in the "AST" must be functions of type `Builder -> Builder`
* A tuple `()` denotes a sequential operation and results in the composition of all functions within it, since its also an element, it compiles to a function that takes a builder and applies the inner composed function.
* A list `[]` denotes a branching operation and results in creating a function that applies the `.branch` method to its argument, since its also an element, it compiles to a function of type `Builder -> BuilderTree`.

To use the DSL you have to import the `tensorbuilder.dsl` module, in patches the `tensorbuilder.tensorbuilder.Builder` class with a `tensorbuilder.tensorbuilder.Builder.pipe` methods that takes an "AST" of such form, compiles it to a function which represents the desired transformation, and finally applies it to the instance `Builder`. The extra argument of `pipe` (\*args) are treated as tuple, so you don't need to include on the first layer.
The `dsl` ships with functions with the same names as all the methods in the `Builder` class, so you get the same API, plus its operations are also chainable so you have to do very little to you code if you want to use the DSL.

Lets see an example, here is the previous example about branching with the the full `patch`, this time using the `dsl` module

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.patch
    import tensorbuilder.dsl as dl #<== Notice the alias

    x = tf.placeholder(tf.float32, shape=[None, 5])
    keep_prob = tf.placeholder(tf.float32)

    h = x.builder().pipe(
        dl.fully_connected(10),
        [
            dl.relu_layer(3)
        ,
            (dl.tanh_layer(9),
            [
              	dl.sigmoid_layer(6)
            ,
                dl
                .dropout(keep_prob)
                .softmax_layer(8)
            ])
        ],
        dl.sigmoid_layer(6)
        .tensor()
    )

    print(h)

As you see a lot of noise is gone, some `dl` terms appeared, and a few `,`s where introduced, but the end result better reveals the structure of you network, plus its very easy to modify.

## Documentation

The main documentaion is [here](http://cgarciae.github.io/tensorbuilder/tensorbuilder.m.html). The documentation for the complete project is [here](http://cgarciae.github.io/tensorbuilder/).

## Examples

Here are the examples for each method of the API. If you are understand all examples, then you've understood the complete API.

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

    # The previous is the same as

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
