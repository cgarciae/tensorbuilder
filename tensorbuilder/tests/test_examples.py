from tensorbuilder.api import *
import tensorflow as tf


class TestExamples(object):


    def test_lambdas(self):

        x, b = tf.placeholder('float'), tf.placeholder('float')

        f = (T + b) / (T + 10)  #lambda x: (x + b) / (x + 10)
        y = f(x)

        assert "div" in y.name


    def test_composition(self):
        x, w, b = tf.placeholder('float', [None, 5]),  tf.placeholder('float', [5, 3]), tf.placeholder('float', [3])

        f = T.matmul(w) >> T + b >> T.sigmoid()
        y = f(x)

        assert "Sigmoid" in y.name

    def test_tf_nn(self):
        x, w, b = tf.placeholder('float', [None, 5]),  tf.placeholder('float', [5, 3]), tf.placeholder('float', [3])

        f = T.matmul(w).add(b).sigmoid()
        y = f(x)

        assert "Sigmoid" in y.name


    def test_layers(self):

        x = tf.placeholder('float', [None, 5])

        y = Pipe(
          x,
          T.layers.fully_connected(64, activation_fn=tf.nn.sigmoid)  # sigmoid layer 64
          .layers.fully_connected(32, activation_fn=tf.nn.tanh)  # tanh layer 32
          .layers.fully_connected(16, activation_fn=None)  # linear layer 16
          .layers.fully_connected(8, activation_fn=tf.nn.relu)  # relu layer 8
        )

        assert "Relu" in y.name