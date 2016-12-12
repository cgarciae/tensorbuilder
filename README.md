# Tensor Builder
TensorBuilder had a mayor refactoring and is now based on [Phi](https://github.com/cgarciae/phi). Updates to the README comming soon!

### Goals
Comming Soon!

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