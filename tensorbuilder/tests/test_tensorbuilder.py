from tensorbuilder import tensorbuilder as tb
import tensorflow as tf

class TestTensorBuilder(object):
    """docstring for TestBuilder"""

    @classmethod
    def setup_method(self):
        self.x = tf.placeholder('float', shape=[None, 5], name='x')
        self.w = tf.transpose(tf.Variable(
            [[1.,2.,3.,4.,5.],
            [6.,7.,8.,9.,10.]]
        ), name='w')
        self.b = tf.Variable(
            [1.,2.],
            name='b'
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

    def test_summaries_patch(self):
        mean = tb.pipe(
            self.x,
            tb.reduce_mean().make_scalar_summary('mean')
        )
        assert "Mean" in mean.name

        mean_summary = tb.pipe(
            self.x,
            tb.reduce_mean().scalar_summary('mean')
        )
        assert "ScalarSummary" in mean_summary.name

        print(tb.make_histogram_summary.__doc__)
