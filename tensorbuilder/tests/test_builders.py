import tensorflow as tf
import tensorbuilder as tb
dl = tb.dl


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
        h2 = tb.build(self.x).unit(h).tensor()

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
        h1 = dl.pipe(
            self.x
            ,
            { tf.device("/cpu:0"):

                dl.softmax()
            }
            ,
            dl.tensor()
        )

        assert "CPU:0" in h1.device


if __name__ == '__main__':
    TestBuilder().test_then_with_1()
    print "pass"
