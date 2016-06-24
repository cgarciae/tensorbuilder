from tensorbuilder import tb
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 5])

def test_empty_pipe():
    builder = tb.pipe(x)

    assert x == builder.tensor()

def test_compile():

    f = tb.compile(
        tb.build, #accept a Tensor as a parameter and create a Builder
        [
            { tf.device("/gpu:0"):
                tb.relu_layer(20)
            }
        ,
            { tf.device("/gpu:1"):
                tb.sigmoid_layer(20)
            }
        ,
            { tf.device("/cpu:0"):
                tb.tanh_layer(20)
            }
        ],
        tb.relu_layer(10)
        .tensor()
    )

    h = f(x)

    assert "Relu" in h.name