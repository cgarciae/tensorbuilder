from tensorbuilder import tensorbuilder as tb
from phi import P, Rec, Obj
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

        matmul, add = tb.ref('matmul'), tb.ref('add')

        y = P(
            self.x,
            tb
            .matmul(self.w).on(matmul)
            .add(self.b).on(add)
            .relu()
        )

        assert "Relu" in y.name
        assert "MatMul" in matmul().name
        assert "Add" in add().name

    def test_summaries_patch(self):
        name = P(
            self.x,
            tb.reduce_mean().make_scalar_summary('mean'),
            Rec.name
        )
        assert "Mean" in name

        name = P(
            self.x,
            tb.reduce_mean().scalar_summary('mean'),
            Rec.name
        )
        assert "ScalarSummary" in name

    def test_layers_patch(self):
        softmax_layer = P(
            self.x,
            tb.layers.sigmoid(10)
            .layers.softmax(20)
        )
        assert "Softmax" in softmax_layer.name

    def test_concat(self):
        concatenated = P(
            self.x,
            [
                tb.layers.softmax(3)
            ,
                tb.layers.tanh(2)
            ,
                tb.layers.sigmoid(5)
            ],
            tb.concat(1)
        )

        assert int(concatenated.get_shape()[1]) == 10

    def test_rnn_utilities(self):
        assert tb.rnn_placeholders_from_state
        assert tb.rnn_state_feed_dict
