# Init code
from tensorbuilder import *
import tensorbuilder
import extras
from dsl import *
import nn
#version
__version__ = "0.0.1"

__all__ = ["tensorbuilder", "dsl", "extras"]

if __name__ == '__main__':
    import tensorflow as tf

    x = tf.placeholder(tf.float32, shape=[None, 5])

    h = x.builder().pipe(
        connect_layer(2)
        .map_softmax(),
        [
            sigmoid_layer(10)
        ,
            [
                tanh_layer(4)
            ,
                tanh_layer(4)
            ]
        ],
        connect_layer(3),
        [
            map_relu()
        ,
            map_softmax()
        ,
            [
                tanh_layer(4)
            ,
                tanh_layer(4)
            ]
        ],
        tensors()
    )

    print(h)
