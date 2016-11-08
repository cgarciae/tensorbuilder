import tensorflow as tf
from tensorbuilder import builder
from fn import _
# from tensorbuilder import tb

add2 = _ + 2
mul3 = _ * 3
get_list = lambda x: [1,2,3]


class TestBuilder(object):
    """docstring for TestBuilder"""

    x = tf.placeholder(tf.float32, shape=[None, 5])

    def test_underscore(self):
        assert builder._(add2)(4) == 6

        assert builder._(add2)._(mul3)(4) == 18

        assert builder.pipe(4, add2, mul3) == 18

        assert builder.pipe(
            4,
            [
            (
                add2,
                mul3
            )
            ,
            (
                mul3,
                add2
            )
            ]
        ) == [18, 14]

        assert builder.pipe(
            4,
            [
                (
                add2,
                mul3
                )
            ,
                [
                    (
                    add2,
                    mul3
                    )
                ,
                    (
                    mul3,
                    add2
                    )
                ]
            ]
        ) == [18, 18, 14]

        assert builder.pipe(
            4,
            [
                (
                add2,
                mul3
                )
            ,
                [
                    (
                    add2,
                    mul3
                    )
                ,
                    (
                    mul3,
                    add2
                    )
                ,
                    get_list
                ]
            ]
        ) == [18, 18, 14, get_list(None)]

        [a, b, c] = builder.pipe(
            4,
            [
                (
                add2,
                mul3
                )
            ,
                [
                    (
                    add2,
                    mul3
                    )
                ,
                    (
                    mul3,
                    add2
                    )
                ]
            ]
        )

        assert a == 18 and b == 18 and c == 14


#
# class TestBuilder(object):
#     """docstring for TestBuilder"""
#
#     x = tf.placeholder(tf.float32, shape=[None, 5])
#
#     def test_build(self):
#         builder = tb.build(self.x)
#         assert type(builder) == tb.Builder
#
#     def test_unit(self):
#         h = tf.nn.softmax(self.x)
#         h2 = tb.build(self.x)._unit(h).tensor()
#
#         assert h2 == h
#
#     def test_branch(self):
#         pass
#
#     def test_then_with_1(self):
#         h1 = (
#             tb.build(self.x)
#             .then_with(tf.device, "/cpu:0")(lambda x:
#                 x.softmax()
#             )
#             .tensor()
#         )
#
#         h2 = (
#             tb.build(self.x)
#             .with_device("/cpu:0")(lambda x:
#                 x.softmax()
#             )
#             .tensor()
#         )
#
#         assert "CPU:0" in h1.device
#         assert "CPU:0" in h2.device
#
#     def test_then_with_2(self):
#         h1 = tb.pipe(
#             self.x
#             ,
#             { tf.device("/cpu:0"):
#
#                 tb.softmax()
#             }
#             ,
#             tb.tensor()
#         )
#
#         assert "CPU:0" in h1.device
#
# class TestBuilderTree(object):
#
#     def test_branches(self):
#         a = tb.build(tf.placeholder(tf.float32, shape=[None, 8]))
#         b = tb.build(tf.placeholder(tf.float32, shape=[None, 8]))
#
#         tree = tb.branches([a, b])
#
#         assert type(tree) == tb.BuilderTree
#
#         [a2, b2] = tree.builders()
#
#         assert a.tensor() == a2.tensor() and b.tensor() == b2.tensor()
#
#
#
# if __name__ == '__main__':
#     TestBuilder().test_then_with_1()
#     print "pass"
