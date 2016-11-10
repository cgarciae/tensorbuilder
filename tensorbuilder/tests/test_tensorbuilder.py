from tensorbuilder import tensorbuilder as tb
import tensorflow as tf

class TestTensorBuilder(object):
    """docstring for TestBuilder"""

    @classmethod
    def setup_method(self):
        self.x = tf.placeholder('float', shape=[None, 5])
        self.w = tf.transpose(tf.Variable(
            [[1.,2.,3.,4.,5.],
            [6.,7.,8.,9.,10.]]
        ))
        self.b = tf.Variable(
            [1.,2.]
        )


    def test_patch(self):
        assert tb.matmul

        y = tb.pipe(
            self.x,
            tb
            .matmul(self.w)
            .add(self.b)
            .relu()
        )

        assert "Relu" in y.name
