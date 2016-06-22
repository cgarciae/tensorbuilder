# DSL

TensorBuilder includes a dsl to make creating complex neural networks even easier. Use it if you are experimenting with complex architectures with branches, otherwise stick to basic API. The DSL has these rules:

* All elements in the "AST" must be functions of type `Builder -> Builder`
* A tuple `()` denotes a sequential operation and results in the composition of all functions within it, since its also an element, it compiles to a function that takes a builder and applies the inner composed function.
* A list `[]` denotes a branching operation and results in creating a function that applies the `.branch` method to its argument, since its also an element, it compiles to a function of type `Builder -> BuilderTree`.

To use the DSL you have to import the `tensorbuilder.dsl` module, in patches the `tensorbuilder.tensorbuilder.Builder` class with a `tensorbuilder.tensorbuilder.Builder.pipe` methods that takes an "AST" of such form, compiles it to a function which represents the desired transformation, and finally applies it to the instance `Builder`. The extra argument of `pipe` (\*args) are treated as tuple, so you don't need to include on the first layer.
The `dsl` ships with functions with the same names as all the methods in the `Builder` class, so you get the same API, plus its operations are also chainable so you have to do very little to you code if you want to use the DSL.

Lets see an example, here is the previous example about branching with the the full `patch`, this time using the `dsl` module

    import tensorflow as tf
    import tensorbuilder as tb
    import tensorbuilder.patch
    import tensorbuilder.dsl as dl #<== Notice the alias

    x = tf.placeholder(tf.float32, shape=[None, 5])
    keep_prob = tf.placeholder(tf.float32)

    h = x.builder().pipe(
        dl.fully_connected(10),
        [
            dl.relu_layer(3)
        ,
            (dl.tanh_layer(9),
            [
              	dl.sigmoid_layer(6)
            ,
                dl
                .dropout(keep_prob)
                .softmax_layer(8)
            ])
        ],
        dl.sigmoid_layer(6)
        .tensor()
    )

    print(h)

As you see a lot of noise is gone, some `dl` terms appeared, and a few `,`s where introduced, but the end result better reveals the structure of you network, plus its very easy to modify.
