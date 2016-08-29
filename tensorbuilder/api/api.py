import builder
import builder_tree
import applicative
from tensorbuilder import tensordata

class API(applicative.Applicative):
    """
    This class represents the public API of TensorBuilder that is created as a class and given to the user as the instance object `tb`. This is done because modules cannot define `__call__` and being function is important for the DSL. `tb` is actually an Applicative that contains the identity function, therefore you can use `tb` as an element inside and expression whenever you need to just return the argument. This is useful because in some cases you create branches in which one just contains the argument unmodified.
    """
    def __init__(self, f):
        super(API, self).__init__(f)

    def build(self, tensor):
        """
        Takes a Tensor and returns a Builder that contians it.

        ** Parameters **

        * `tensor`: a tensorflow Tensor


        #### Example
        The following example shows you how to construct a `tensorbuilder.tensorbuilder.Builder` from a tensorflow Tensor.

            import tensorflow as tf
            import tensorbuilder as tb

            a = tf.placeholder(tf.float32, shape=[None, 8])
            a_builder = tb.build(a)
        """
        return self.Builder(tensor)

    def branches(self, builder_iterable):
        """
        Takes an iterable with elements of type `Builder` or `BuilderTree` and returns a `BuilderTree`

        ** Parameters **

        * `builder_iterable`: list of type `iterable( Builder | BuilderTree)`

        #### Example
        Given a list of Builders and/or BuilderTrees you construct a `tensorbuilder.tensorbuilder.BuilderTree` like this

            import tensorflow as tf
            import tensorbuilder as tb

            a = tb.build(tf.placeholder(tf.float32, shape=[None, 8]))
            b = tb.build(tf.placeholder(tf.float32, shape=[None, 12]))

            tree = tb.branches([a, b])

        > **Note:** Ideally `BuilderTree`s are constructed using `Builder.branch`.
        """
        return self.BuilderTree(builder_iterable)

API.data = tensordata.Data

API.Builder = builder.Builder
API.BuilderTree = builder_tree.BuilderTree
API.Applicative = applicative.Applicative
