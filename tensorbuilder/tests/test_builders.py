import tensorflow as tf
from tensorbuilder import tb

def func(x):
    return x + 1

class TestBuilder(object):
    """docstring for TestBuilder"""

    x = tf.placeholder(tf.float32, shape=[None, 5])

    def test_build(self):
        builder = tb.build(self.x)
        assert type(builder) == tb.Builder

    def test_unit(self):
        h = tf.nn.softmax(self.x)
        h2 = tb.build(self.x)._unit(h).tensor()

        assert h2 == h

    def test_branch(self):
        pass

    def test_then_with_1(self):
        h1 = (
            tb.build(self.x)
            .then_with(tf.device, "/cpu:0")(lambda x:
                x.softmax()
            )
            .tensor()
        )

        h2 = (
            tb.build(self.x)
            .with_device("/cpu:0")(lambda x:
                x.softmax()
            )
            .tensor()
        )

        assert "CPU:0" in h1.device
        assert "CPU:0" in h2.device

    def test_then_with_2(self):
        h1 = tb.pipe(
            self.x
            ,
            { tf.device("/cpu:0"):

                tb.softmax()
            }
            ,
            tb.tensor()
        )

        assert "CPU:0" in h1.device

class TestBuilderTree(object):

    def test_branches(self):
        a = tb.build(tf.placeholder(tf.float32, shape=[None, 8]))
        b = tb.build(tf.placeholder(tf.float32, shape=[None, 8]))

        tree = tb.branches([a, b])

        assert type(tree) == tb.BuilderTree

        [a2, b2] = tree.builders()

        assert a.tensor() == a2.tensor() and b.tensor() == b2.tensor()



if __name__ == '__main__':
    TestBuilder().test_then_with_1()
    print "pass"
