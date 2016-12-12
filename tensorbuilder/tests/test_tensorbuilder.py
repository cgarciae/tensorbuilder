from tensorbuilder import T
from phi import P, Rec
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

        matmul, add = T.Ref('matmul'), T.Ref('add')

        y = T.Pipe(
            self.x,

            T
            .matmul(self.w).Write(matmul)
            .add(self.b).Write(add)
            .relu()
        )

        assert "Relu" in y.name
        assert "MatMul" in matmul().name
        assert "Add" in add().name

    def test_summaries_patch(self):
        name = T.Pipe(
            self.x,
            T.reduce_mean().make_scalar_summary('mean'),
            Rec.name
        )
        assert "Mean" in name

        name = T.Pipe(
            self.x,
            T.reduce_mean().scalar_summary('mean'),
            Rec.name
        )
        assert "ScalarSummary" in name

    def test_layers_patch(self):
        softmax_layer = T.Pipe(
            self.x,
            T
            .sigmoid_layer(10)
            .softmax_layer(20)
        )
        assert "Softmax" in softmax_layer.name

    def test_concat(self):
        concatenated = T.Pipe(
            self.x,
            [
                T.softmax_layer(3)
            ,
                T.tanh_layer(2)
            ,
                T.sigmoid_layer(5)
            ],
            T.concat(1)
        )

        assert int(concatenated.get_shape()[1]) == 10

    def test_rnn_utilities(self):
        assert T.rnn_placeholders_from_state
        assert T.rnn_state_feed_dict
