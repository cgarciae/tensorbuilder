import tensorflow as tf
from tensorflow.python.framework import tensor_shape

builders_blacklist = [
    "sampled_softmax_loss"
]

def _polynomial(tensor):
    size = int(tensor.get_shape()[1])
    pows = [ tf.pow(tensor[:, n], n + 1) for n in range(size) ]
    return tf.transpose(tf.pack(pows))

def polynomial_layer(builder, size, **kwargs):
    """
    Creates a fully connected layer of size `size` and then applies the activation function
    ```
    y(i) = z(i)^(i+1)
    ```
    where `z = w*x + b`
    """
    with tf.variable_scope("polynomial_layer"):
        return builder.fully_connected(size, activation_fn=_polynomial, **kwargs)

def minimize(tensor, optimizer, *args, **kwargs):
    return optimizer.minimize(tensor, *args, **kwargs)

def maximize(tensor, optimizer, *args, **kwargs):
    return optimizer.maximize(tensor, *args, **kwargs)

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

def ensamble_dropout(tree, keep_prob, seed=None, name=None):
    with tf.op_scope(tree.tensors(), name, "drop_layer"):
        return tree.map_each(drop_layer, keep_prob, seed=seed, name=name)

def add_regularization_loss(tensor, graph=None, scope='add_regularization_loss'):
    if not graph:
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    else:
        reg_losses = graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    with tf.variable_scope(scope):
        reg_loss = tf.reduce_sum([ tf.reduce_mean(reg_loss) for reg_loss in reg_losses ], name='reg_loss_mean_sum')
        return tf.add(tensor, reg_loss)


from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages


@add_arg_scope
def fully_connected_from_product(inputs,
                    activation_fn=None,
                    normalizer_fn=None,
                    normalizer_params=None,
                    biases_initializer=init_ops.zeros_initializer,
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):
  """Adds a fully connected layer.
  `fully_connected` creates a variable called `weights`, representing a fully
  connected weight matrix, which is multiplied by the `inputs` to produce a
  `Tensor` of hidden units. If a `normalizer_fn` is provided (such as
  `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
  None and a `biases_initializer` is provided then a `biases` variable would be
  created and added the hidden units. Finally, if `activation_fn` is not `None`,
  it is applied to the hidden units as well.
  Note: that if `inputs` have a rank greater than 2, then `inputs` is flattened
  prior to the initial matrix multiply by `weights`.
  Args:
    inputs: A tensor of with at least rank 2 and value for the last dimension,
      i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
    num_outputs: Integer, the number of output units in the layer.
    activation_fn: activation function.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalize_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collections per variable.
    outputs_collections: collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for variable_op_scope.
  Returns:
     the tensor variable representing the result of the series of operations.
  Raises:
    ValueError: if x has rank less than 2 or if its last dimension is not set.
  """

  with variable_scope.variable_op_scope([inputs],
                                        scope,
                                        'fully_connected_from_product',
                                        reuse=reuse) as sc:

    outputs = inputs
    num_outputs = int(outputs.get_shape()[1])
    dtype = outputs.dtype

    if normalizer_fn:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs,],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          collections=biases_collections,
                                          trainable=trainable)
        outputs = nn.bias_add(outputs, biases)
    if activation_fn:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@tf.contrib.framework.add_arg_scope
def sampled_softmax_loss(
                    inputs,
                    num_outputs,
                    labels,
                    num_sampled,
                    sampled_softmax_loss_args={},
                    activation_fn=None,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=tf.contrib.layers.initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=tf.zeros_initializer,
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):
  """Adds a fully connected layer.
  `fully_connected` creates a variable called `weights`, representing a fully
  connected weight matrix, which is multiplied by the `inputs` to produce a
  `Tensor` of hidden units. If a `normalizer_fn` is provided (such as
  `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
  None and a `biases_initializer` is provided then a `biases` variable would be
  created and added the hidden units. Finally, if `activation_fn` is not `None`,
  it is applied to the hidden units as well.
  Note: that if `inputs` have a rank greater than 2, then `inputs` is flattened
  prior to the initial matrix multiply by `weights`.
  Args:
    inputs: A tensor of with at least rank 2 and value for the last dimension,
      i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
    num_outputs: Integer, the number of output units in the layer.
    activation_fn: activation function.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalize_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collections per variable.
    outputs_collections: collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for variable_op_scope.
  Returns:
     the tensor variable representing the result of the series of operations.
  Raises:
    ValueError: if x has rank less than 2 or if its last dimension is not set.
  """
  if not isinstance(num_outputs, int):
    raise ValueError('num_outputs should be integer, got %s.', num_outputs)
  with tf.variable_op_scope([inputs],
                            scope,
                            'sampled_softmax_loss',
                            reuse=reuse) as sc:
    inputs = tf.convert_to_tensor(inputs)
    dtype = inputs.dtype.base_dtype
    inputs_shape = inputs.get_shape()
    num_input_units = tf.contrib.layers.utils.last_dimension(inputs_shape, min_rank=2)

    static_shape = inputs_shape.as_list()
    static_shape[-1] = num_outputs

    weights_shape = [num_input_units, num_outputs]
    weights_collections = tf.contrib.layers.utils.get_variable_collections(
        variables_collections, 'weights')
    weights = tf.contrib.framework.model_variable('weights',
                                       shape=weights_shape,
                                       dtype=dtype,
                                       initializer=weights_initializer,
                                       regularizer=weights_regularizer,
                                       collections=weights_collections,
                                       trainable=trainable)
    if len(static_shape) > 2:
      # Reshape inputs
      inputs = tf.reshape(inputs, [-1, num_input_units])


    biases_collections = tf.contrib.layers.utils.get_variable_collections(
        variables_collections, 'biases')
    biases = tf.contrib.framework.model_variable('biases',
                                      shape=[num_outputs,],
                                      dtype=dtype,
                                      initializer=biases_initializer,
                                      regularizer=biases_regularizer,
                                      collections=biases_collections,
                                      trainable=trainable)


    outputs = tf.nn.sampled_softmax_loss(
        tf.transpose(weights), biases, inputs,
        labels, num_sampled, num_outputs,
        **sampled_softmax_loss_args)

    return tf.contrib.layers.utils.collect_named_outputs(outputs_collections, sc.name, outputs)

def patch_classes(Builder, BuilderTree, Applicative):
    #Builder
    Builder.register_method(polynomial_layer, "tensorbuilder.Builder")
    Builder.register_map_method(minimize, "tensorbuilder.Builder")
    Builder.register_map_method(maximize, "tensorbuilder.Builder")
    Builder.register_map_method(drop_layer, "tensorbuilder.Builder")
    Builder.register_map_method(add_regularization_loss, "tensorbuilder")
    Builder.register_map_method(fully_connected_from_product, "tensorbuilder")
    Builder.register_map_method(sampled_softmax_loss, "tensorbuilder")
    Builder.register_map_method(sampled_softmax_loss, "tensorbuilder")


    #Tree
    BuilderTree.register_method(ensamble_dropout, "tensorbuilder.BuilderTree")
