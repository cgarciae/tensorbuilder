import tensorflow as tf
import tensorbuilder as tb
import tensorbuilder.patches.tensorflow.patch
import tensorbuilder.dsl as dl


x = tf.placeholder(tf.float32, shape=[None, 5])
keep_prob = tf.placeholder(tf.float32)

h = (
	x.builder()
	.tanh_layer(10)
	.dropout(keep_prob)
	.softmax_layer(3)
	.tensor()
)

h2 = x.builder().pipe(
    dl.tanh_layer(10)
    .dropout(keep_prob)
    .softmax_layer(3)
    .tensor()
)

print h
print h2
