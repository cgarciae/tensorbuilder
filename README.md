# Tensor Builder
TensorBuilder is a TensorFlow-based library that enables you to easily create complex neural networks using functional programming.

##Import

For demonstration purposes we will import right now everything we will need for the rest of the exercises like this
```python
from tensorbuilder.api import *
import tensorflow as tf
```
but you can also import just what you need from the `tensorbuilder` module.

## Phi
#### Lambdas
With the `T` object you can create quick math-like lambdas using any operator, this lets you write things like
```python
x, b = tf.placeholder('float'), tf.placeholder('float')

f = (T + b) / (T + 10)  #lambda x: (x + b) / (x + 10)
y = f(x)

assert "div" in y.name
```

#### Composition
Use function composition with the `>>` operator to improve readability
```python
x, w, b = tf.placeholder('float', [None, 5]),  tf.placeholder('float', [5, 3]), tf.placeholder('float', [3])

f = T.matmul(w) >> T + b >> T.sigmoid()
y = f(x)

assert "Sigmoid" in y.name
```

## tf + nn
Any function from the `tf` and `nn` modules is a method from the `T` object, as before you can use the `>>` operator or you can chain them to produce complex functions
```python
x, w, b = tf.placeholder('float', [None, 5]),  tf.placeholder('float', [5, 3]), tf.placeholder('float', [3])

f = T.matmul(w).add(b).sigmoid()
y = f(x)

assert "Sigmoid" in y.name
```
## layers
#### affine
You can use functions from the `tf.contrib.layers` module via the `T.layers` property. Here we will use [Pipe](https://github.com/cgarciae/phi#seq-and-pipe) to apply a value directly to an expression:
```python
x = tf.placeholder('float', [None, 5])

y = Pipe(
  x,
  T.layers.fully_connected(64, activation_fn=tf.nn.sigmoid)  # sigmoid layer 64
  .layers.fully_connected(32, activation_fn=tf.nn.tanh)  # tanh layer 32
  .layers.fully_connected(16, activation_fn=None)  # linear layer 16
  .layers.fully_connected(8, activation_fn=tf.nn.relu)  # relu layer 8
)

assert "Relu" in y.name
```
However, since it is such a common task to build fully_connected layers using the different functions from the `tf.nn` module, we've (dynamically) create all combination of these as their own methods so you con rewrite the previous as
```python
x = tf.placeholder('float', [None, 5])

y = Pipe(
  x,
  T.sigmoid_layer(64)  # sigmoid layer 64
  .tanh_layer(32)  # tanh layer 32
  .linear_layer(16)  # linear layer 16
  .relu_layer(8)  # relu layer 8
)

assert "Relu" in y.name
```
The latter is much more compact, English readable, and reduces a lot of noise.

#### convolutional
Coming soon!

## leveraging phi
Coming soon!

## summary
Coming soon!

## other ops
Coming soon!

## Installation
Tensor Builder assumes you have a working `tensorflow` installation. We don't include it in the `requirements.txt` since the installation of tensorflow varies depending on your setup.

#### From pypi
```
pip install tensorbuilder
```

#### From github
For the latest development version
```
pip install git+https://github.com/cgarciae/tensorbuilder.git@develop
```

## Getting Started

Create neural network with a [5, 10, 3] architecture with a `softmax` output layer and a `tanh` hidden layer through a Builder and then get back its tensor:

```python
import tensorflow as tf
from tensorbuilder import T

x = tf.placeholder(tf.float32, shape=[None, 5])
keep_prob = tf.placeholder(tf.float32)

h = T.Pipe(
  x,
  T.tanh_layer(10) # tanh(x * w + b)
  .dropout(keep_prob) # dropout(x, keep_prob)
  .softmax_layer(3) # softmax(x * w + b)
)
```

## Features
Comming Soon!

## Documentation
Comming Soon!

## The Guide
Comming Soon!

## Full Example
Next is an example with all the features of TensorBuilder including the DSL, branching and scoping. It creates a branched computation where each branch is executed on a different device. All branches are then reduced to a single layer, but the computation is the branched again to obtain both the activation function and the trainer.

```python
import tensorflow as tf
from tensorbuilder import T

x = placeholder(tf.float32, shape=[None, 10])
y = placeholder(tf.float32, shape=[None, 5])

[activation, trainer] = T.Pipe(
    x,
    [
        T.With( tf.device("/gpu:0"):
            T.relu_layer(20)
        )
    ,
        T.With( tf.device("/gpu:1"):
            T.sigmoid_layer(20)
        )
    ,
        T.With( tf.device("/cpu:0"):
            T.tanh_layer(20)
        )
    ],
    T.linear_layer(5),
    [
        T.softmax() # activation
    ,
        T
        .softmax_cross_entropy_with_logits(y) # loss
        .minimize(tf.train.AdamOptimizer(0.01)) # trainer
    ]
)
```