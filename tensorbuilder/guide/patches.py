"""
Since the Builder class is mostly a [monadic](http://stackoverflow.com/a/194207/2118130) structure which helps you build the computation, in the interest of letting other libraries use Tensor Builder to obtain a fluid API + DSL, TensorBuilder ships the Builder class with no Tensor specific methods but instead contains helpers which enable you to register external functions as methods. Library authors are encouraged to create patches so they can worry about the basic operations and let TensorBuilder add the syntax.

However, TensorBuilder ships with its own main patch, the `tensorbuilder.patch` which adds methods focused on helping you to easily create complex network, it does so (mostly) by registering (cherry picking) methods from other libraries. The intention here is get you the best of what is out there. Here is the list of all the patches you can use.

* `import tensorbuilder.patch`
* `import tensorbuilder.slim_patch`
* `import tensorbuilder.patches.tensorflow.patch`
* `import tensorbuilder.patches.tensorflow.slim`
* `import tensorbuilder.patches.tflearn.patch`

Check out is an example using the `tflearn` patch

    import tflearn
    import tensorbuilder as tb
    import tensorbuilder.patches.tflearn.patch

    model = (
    	tflearn.input_data(shape=[None, 784]).builder()
    	.fully_connected(64)
    	.dropout(0.5)
    	.fully_connected(10, activation='softmax')
    	.regression(optimizer='adam', loss='categorical_crossentropy')
    	.map(tflearn.DNN)
    	.tensor()
    )

    print(model)
"""
