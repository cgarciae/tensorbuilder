import tensorflow as tf

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

def patch_classes(Builder, BuilderTree, Applicative):
    Builder.register_method(polynomial_layer, "tensorbuilder")
    Builder.register_map_method(minimize, "tensorbuilder")
    Builder.register_map_method(maximize, "tensorbuilder")
