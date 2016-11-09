import tensorflow as tf
from tensorbuilder import Builder
from fn import _

builder = Builder()
# from tensorbuilder import tb

add2 = _ + 2
mul3 = _ * 3
get_list = lambda x: [1,2,3]
a2_plus_b_minus_2c = lambda a, b, c: a ** 2 + b - 2*c


@builder.register("test.lib")
def add(a, b):
    """Some docs"""
    return a + b

@builder.register2("test.lib")
def pow(a, b):
    return a ** b

@builder.register_method("test.lib")
def get_function_name(builder):
    return builder.f.__name__



class TestBuilder(object):
    """docstring for TestBuilder"""

    @classmethod
    def setup_method(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 5])

    def test_underscore(self):
        assert builder._(add2)(4) == 6
        assert builder._(add2)._(mul3)(4) == 18

    def test_underscores(self):
        assert builder._(a2_plus_b_minus_2c, 2, 4)(3) == 3 # (3)^2 + 2 - 2*4
        assert builder._2(a2_plus_b_minus_2c, 2, 4)(3) == -1 # (2)^2 + 3 - 2*4
        assert builder._3(a2_plus_b_minus_2c, 2, 4)(3) == 2 # (2)^2 + 4 - 2*3

    def test_pipe(self):
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

    def test_scope(self):
        y = builder.ref()

        z = builder.pipe(
            self.x,
            { tf.name_scope('TEST'):
                (
                _ * 2,
                _ + 4,
                builder.store(y)
                )
            },
            _ ** 3
        )

        assert "TEST/" in y().name
        assert "TEST/" not in z.name

    def test_register(self):

        #register
        assert 5 == builder.pipe(
            3,
            builder.add(2)
        )

        #register2
        assert 8 == builder.pipe(
            3,
            builder.pow(2)
        )

        #register_method
        assert "_identity" == builder.get_function_name()

    def test_using_run(self):
        assert 8 == builder.using(3).add(2).add(3).run()

    def test_reference(self):
        ref = builder.ref()

        assert 8 == builder.using(3).add(2).store(ref).add(3).run()
        assert 5 == ref()

    def test_ref_props(self):

        a = builder.ref()
        b = builder.ref()

        assert [7, 3, 5] == builder.pipe(
            1,
            add2, a.set,
            add2, b.set,
            add2,
            [
                builder.identity,
                a,
                b
            ]
        )



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
