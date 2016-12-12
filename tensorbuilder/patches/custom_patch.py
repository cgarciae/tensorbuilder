import tensorflow as tf
from tensorbuilder import T, TensorBuilder
from phi import P, utils

@TensorBuilder.Register1("tb")
def inception_layer(tensor, num_outputs, **kwargs):
    stride = 1
    pool_operation = T.max_pool2d
    pool_kernel = [3, 3]
    scope = None
    reuse = None

    if 'stride' in kwargs:
        stride = kwargs['stride']
    else:
        kwargs['stride'] = stride


    if 'pool_kernel' in kwargs:
        pool_kernel = kwargs['pool_kernel']
        del kwargs['pool_kernel']

    if 'scope' in kwargs:
        scope = kwargs['scope']
        del kwargs['scope']

    if 'reuse' in kwargs:
        reuse = kwargs['reuse']
        del kwargs['reuse']

    if 'pool_operation' in kwargs:
        pool_operation = kwargs['pool_operation']
        del kwargs['pool_operation']

    kwargs_no_stride = kwargs.copy()
    del kwargs_no_stride['stride']

    with tf.variable_scope(scope, default_name='InceptionLayer', reuse=reuse):
        return P(
            tensor,
            [
                T.convolution2d(num_outputs, [1, 1], **dict(kwargs, scope="Conv1x1"))
            ,
                P.With( T.variable_scope("Branch3x3"),
                    T
                    .convolution2d(num_outputs, [1, 1], **dict(kwargs_no_stride, scope="Conv1x1"))
                    .convolution2d(num_outputs, [3, 3], **dict(kwargs, scope="Conv3x3"))
                )
            ,
                P.With( T.variable_scope("Branch5x5"),
                    T
                    .convolution2d(num_outputs, [1, 1], **dict(kwargs_no_stride, scope="Conv1x1"))
                    .convolution2d(num_outputs, [5, 5], **dict(kwargs, scope="Conv5x5"))
                )
            ,
                P.With( T.variable_scope("BranchPool"),
                    pool_operation(pool_kernel, stride=stride, padding='SAME'),
                    T.convolution2d(num_outputs, [1, 1], **dict(kwargs_no_stride, scope="Conv1x1"))
                )
            ],
            T.concat(3)
        )

@TensorBuilder.Register1("tb")
def minimize(tensor, optimizer, *args, **kwargs):
    return optimizer.minimize(tensor, *args, **kwargs)

@TensorBuilder.Register1("tb")
def maximize(tensor, optimizer, *args, **kwargs):
    return optimizer.maximize(tensor, *args, **kwargs)

@TensorBuilder.Register1("tb")
def drop_layer(x, keep_prob, seed=None, name=None):
  """Computes dropout.
  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.

  Args:
    x: A tensor.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for this operation (optional).
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]`.
  """
  with tf.op_scope([x], name, "drop_layer") as name:
    x = tf.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, float) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = tf.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    noise_shape = [ tf.shape(x)[0], 1 ]
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(
        noise_shape,
        seed=seed,
        dtype=x.dtype
    )

    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = tf.floor(random_tensor)
    ret = x * tf.inv(keep_prob) * binary_tensor
    ret.set_shape(x.get_shape())
    return ret

@TensorBuilder.Register1("tb")
def ensamble_dropout(tree, keep_prob, seed=None, name=None):
    with tf.op_scope(tree.tensors(), name, "drop_layer"):
        return tree.map_each(drop_layer, keep_prob, seed=seed, name=name)

@TensorBuilder.Register1("tb")
def add_regularization_loss(tensor, graph=None, scope='add_regularization_loss'):
    if not graph:
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    else:
        reg_losses = graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    with tf.variable_scope(scope):
        reg_loss = tf.reduce_sum([ tf.reduce_mean(reg_loss) for reg_loss in reg_losses ], name='reg_loss_mean_sum')
        return tf.add(tensor, reg_loss)
