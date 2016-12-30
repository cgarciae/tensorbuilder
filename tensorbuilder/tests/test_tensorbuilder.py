from tensorbuilder.api import *
import tensorflow as tf
from tensorflow.contrib import layers

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

        matmul, add, relu = T.Pipe(
            self.x, T
            .Write(
                matmul = T.matmul(self.w)
            ).Write(
                add = T.add(self.b)
            ).Write(
                relu = T.relu()
            )
            .ReadList('matmul', 'add', 'relu')
        )

        assert "Relu" in relu.name
        assert "MatMul" in matmul.name
        assert "Add" in add.name

    def test_summaries_patch(self):
        name = T.Pipe(
            self.x,
            T.reduce_mean().summary.create_scalar('mean'),
            Rec.name
        )
        assert "Mean" in name

        name = T.Pipe(
            self.x, T
            .reduce_mean().summary.scalar('mean'),
            Rec.name
        )
        assert "mean" in name

    def test_layers_patch(self):
        softmax_layer = T.Pipe(
            self.x, T
            .sigmoid_layer(10)
            .softmax_layer(20)
        )
        assert "Softmax" in softmax_layer.name

    def test_concat(self):
        concatenated = T.Pipe(
            self.x, T
            .List(
                T.softmax_layer(3)
            ,
                T.tanh_layer(2)
            ,
                T.sigmoid_layer(5)
            )
            .concat(1)
        )

        assert int(concatenated.get_shape()[1]) == 10

    def test_rnn_utilities(self):
        assert T.rnn_placeholders_from_state
        assert T.rnn_state_feed_dict


    def test_convolution(self):
        y = Pipe(
            tf.placeholder(tf.float32, shape=[None, 30, 30, 1]),
            T.convolution2d(64, [3,3], activation_fn=tf.nn.elu, normalizer_fn=layers.batch_norm)
        )

        assert "Conv" in y.name and "Elu" in y.name

    def test_inception(self):

        y = Pipe(
            tf.placeholder(tf.float32, shape=[None, 30, 30, 1]),
            T.inception_layer(64, activation_fn=tf.nn.elu, normalizer_fn=layers.batch_norm)
        )

        assert y.name


    def test_inception_net(self):

        with tf.name_scope('inputs'):
            x = tf.placeholder('float', shape=[None, 28, 28, 1], name='x')
            y = tf.placeholder('float', shape=[None, 10], name='y')
            learning_rate = tf.placeholder('float', name='learning_rate')
            keep_prob = tf.placeholder('float', name='keep_prob')


        [h, trainer, loss] = T.Pipe(
            x, T
            .inception_layer(4, activation_fn=tf.nn.elu, normalizer_fn=layers.batch_norm)
            .inception_layer(8, activation_fn=tf.nn.elu, normalizer_fn=layers.batch_norm)
            .elu_conv2d_layer(64, [3, 3], normalizer_fn=layers.batch_norm)
            .max_pool2d(2)
            .elu_conv2d_layer(128, [3, 3], normalizer_fn=layers.batch_norm)
            .max_pool2d(2)
            .elu_conv2d_layer(128, [3, 3], padding='VALID', normalizer_fn=layers.batch_norm)
            .elu_conv2d_layer(128, [3, 3], padding='VALID', normalizer_fn=layers.batch_norm)
            .elu_conv2d_layer(256, [3, 3], padding='VALID', normalizer_fn=layers.batch_norm)
            .flatten()
            .dropout(keep_prob)
            .linear_layer(10) #, scope='logits'),
            .List(
                T.softmax(name='h')
            ,
                With( tf.name_scope('loss'),
                    T.softmax_cross_entropy_with_logits(y).reduce_mean().summary.create_scalar('loss').Write('loss')
                )
                .minimize(tf.train.AdadeltaOptimizer(learning_rate))
            ,
                Read('loss')
            )
        )

        assert "loss" in loss.name

    def test_trainop(self):
        x = tf.placeholder('float', [100, 2])
        y = tf.placeholder('float', [100, 1])


        assert T.TrainOp

        trainer = Pipe(
            x, T
            .sigmoid_layer(2)
            .linear_layer(1).sigmoid_cross_entropy_with_logits(y)

            .TrainOp(tf.train.AdamOptimizer(0.01))
            .Trainer()
        )

        assert trainer
