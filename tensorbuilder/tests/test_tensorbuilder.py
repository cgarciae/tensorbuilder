import ipdb
from tensorbuilder import tensorbuilder as tb
from phi import ph, Rec
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

        matmul, add = tb.Ref('matmul'), tb.Ref('add')

        y = tb.Pipe(
            self.x,

            tb
            .matmul(self.w).On(matmul)
            .add(self.b).On(add)
            .relu()
        )



        assert "Relu" in y.name
        assert "MatMul" in matmul().name
        assert "Add" in add().name

    def test_summaries_patch(self):
        name = tb.Pipe(
            self.x,
            tb.reduce_mean().make_scalar_summary('mean'),
            Rec.name
        )
        assert "Mean" in name

        name = tb.Pipe(
            self.x,
            tb.reduce_mean().scalar_summary('mean'),
            Rec.name
        )
        assert "ScalarSummary" in name

    def test_layers_patch(self):
        softmax_layer = tb.Pipe(
            self.x,
            tb
            .sigmoid_layer(10)
            .softmax_layer(20)
        )
        assert "Softmax" in softmax_layer.name

    def test_concat(self):
        concatenated = tb.Pipe(
            self.x,
            [
                tb.softmax_layer(3)
            ,
                tb.tanh_layer(2)
            ,
                tb.sigmoid_layer(5)
            ],
            tb.concat(1)
        )

        assert int(concatenated.get_shape()[1]) == 10

    def test_rnn_utilities(self):
        assert tb.rnn_placeholders_from_state
        assert tb.rnn_state_feed_dict
