# Basics
Here we will cover the basics of Tensor Builder, for this we will solve one of the simplest classical examples in the history of neural network: the XOR.

We will assume that you have already installed TensorBuilder, if not click [here](https://cgarciae.gitbooks.io/tensorbuilder/content/). Remember that you must have a working installation of TensorFlow.

## Setup
First we will setup our imports, you'll need to have `numpy` installed.

```python
import tensorflow as tf
import numpy as np
from tensorbuilder import tb
```

As you see `tb` is **not** an alias for the `tensorbuilder` module, its an object that we import from this library. There are several reason behind this, one is that implementing it this way reduced a lot of code internally, but it also plays better with the DSL as you might see later.

> **Note:** `tb` is of type `Applicative` and all of its methods are immutable, so down worry about "breaking" it.

Next we create our data and placeholders

```python
#TRUTH TABLE (DATA)
X =     [[0.0,0.0]]; Y =     [[0.0]]
X.append([1.0,0.0]); Y.append([1.0])
X.append([0.0,1.0]); Y.append([1.0])
X.append([1.0,1.0]); Y.append([0.0])

X = np.array(X)
Y = np.array(Y)

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])
```

## The Neural Network
If you are used to creating the `tf.Variable` and writing the layer functions by yourself, don't blink because TensorBuilder is pretty terse, this next code creates both the `activation` function and the `trainer`

```python
[activation, trainer] = (
    tb.build(x)
    .sigmoid_layer(2)
    .linear_layer(1)
    .branch(lambda logit:
    [
        logit.sigmoid() # activation
    ,
        logit
        .sigmoid_cross_entropy_with_logits(y) # loss
        .map(tf.train.AdamOptimizer(0.01).minimize) # trainer
    ])
    .tensors()
)
```

So lets review what is happening, in the first 3 lines you are

1. `tb.build(x)` creating a builder from the input Tensor `x`
2. `.sigmoid_layer(2)` connecting a sigmoid layer with 2 neurons
3. `.linear_layer(1)` connecting a linear layer with 1 neuron

Up to this point we have successfully created the basic architecture of our network, we are only missing passing this result through a `sigmoid` function. But here comes the funny part, we create a branch from a `linear_layer` because

* We want to get both the trainer and the activation function using the fluent API, therefore we have to return two objects
* The function `tf.nn.sigmoid_cross_entropy_with_logits` is defined to work with a logit (`z = w x + b`), therefore we have to apply the `sigmoid` function ourself through in the first branch, and calculate the loss function and trainer in the other. With other loss functions we can apply the activation function before branching, but we still have to branch to return 2 tensors.

The `Builder` class which you've been using doesn't have methods to create trainers, however it this case there is a neat solution which is to create a Trainer in-line using `tf.train`

```python
tf.train.AdamOptimizer(0.01)
```

and then calling passing it minimize method to `.map`. What this does is that it passes the `loss` tensor created by `.sigmoid_cross_entropy_with_logits(y)` as an argument to the `minimize` methods, thus resulting in a tensor computation which minimizes the loss with the Adam trainer.

Finally, the `.tensors()` method at the end gives us back a list of tensor from the branches (it even works with nested branches) and we use pattern matching to assign them to the variables `activation` and `trainer`

 ```
 [activation, trainer] = (...)
 ```

## Training
Now that we have our training operation, we use regular TensorFlow operations to trainer the network. We will train for 2000 epochs using full batch training.

```python
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(2000):
    sess.run(trainer, feed_dict={x: X, y: Y})
```

Finally lets see what our network has learned by running the activation function

```python
for i in range(len(X)):
    print "{0} ==> {1}".format(X[i], sess.run(activation, feed_dict={x: X[i:i+1,:]}))
```




