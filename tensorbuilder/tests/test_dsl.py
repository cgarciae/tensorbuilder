from tensorbuilder import tb
import tensorflow as tf
from tensorbuilder import dl

x = tf.placeholder(tf.float32, shape=[None, 5])

def test_empty_pipe():
    builder = dl.pipe(x)

    assert x == builder.tensor()