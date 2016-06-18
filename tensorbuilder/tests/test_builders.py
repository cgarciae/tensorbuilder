import tensorflow as tf
import tensorbuilder as tb


def func(x):
    return x + 1

class TestBuilder(object):
    """docstring for TestBuilder"""

    x = tf.placeholder(tf.float32, shape=[None, 5])

    def test_build(self):
        builder = tb.build(self.x)
        assert type(builder) == tb.Builder

    def test_unit(self):
        h = tf.nn.softmax(self.x)
        h2 = tb.build(self.x).unit(h).tensor()

        assert h2 == h
