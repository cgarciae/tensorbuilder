import tensorflow as tf
from tensorbuilder.builder import Builder, P, C, M, _0, _1, _2, _3, _4, _5
from fn import _
import math


bl = Builder()
# from tensorbuilder import tb

add2 = _ + 2
mul3 = _ * 3
get_list = lambda x: [1,2,3]
a2_plus_b_minus_2c = lambda a, b, c: a ** 2 + b - 2*c


@bl.register_1("test.lib")
def add(a, b):
    """Some docs"""
    return a + b

@bl.register_2("test.lib")
def pow(a, b):
    return a ** b

@bl.register_method("test.lib")
def get_function_name(bl):
    return bl.f.__name__

class DummyContext:
    def __init__(self, val):
        self.val = val

    def __enter__(self):
        return self.val
    def __exit__(self, type, value, traceback):
        pass


class TestBuilder(object):
    """docstring for TestBuilder"""

    @classmethod
    def setup_method(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 5])

    def test_underscore_1(self):
        assert bl._1(add2)(4) == 6
        assert bl._1(add2)._1(mul3)(4) == 18

        assert bl._(add2)(4) == 6
        assert bl._(add2)._(mul3)(4) == 18

    def test_methods(self):
        assert 9 == P(
            "hello world !!!",
            M.split(" ")
            .filter(M.contains("wor").Not())
            .map(len),
            sum,
            _ + 0.5,
            round
        )

        assert not P(
            [1,2,3],
            M.contains(5)
        )

        class A(object):
            def something(self, x):
                return "y" * x

        assert "yyyy" == P(
            A(),
            M.proxy('something', 4) #registered something
        )

        assert "yy" == P(
            A(),
            M.something(2) #used something
        )

    def test_rrshift(self):
        builder = C(
            _ + 1,
            _ * 2,
            _ + 4
        )
        print(builder)
        print(builder(2))
        assert 10 == 2 >> builder

    def test_0(self):
        from datetime import datetime
        import time

        t0 = datetime.now()

        time.sleep(0.01)

        t1 = 2 >> C(
            _ + 1,
            _0(datetime.now)
        )

        assert t1 > t0

    def test_1(self):
        assert 9 == 2 >> C(
            _ + 1,
            _1(math.pow, 2)
        )

    def test_2(self):
        assert [2, 4] == [1, 2, 3] >> C(
            _2(map, _ + 1),
            _2(filter, _ % 2 == 0)
        )

        assert [2, 4] == P(
            [1, 2, 3],
            _2(map, _ + 1),
            _2(filter, _ % 2 == 0)
        )


    def test_underscores(self):
        assert bl._1(a2_plus_b_minus_2c, 2, 4)(3) == 3 # (3)^2 + 2 - 2*4
        assert bl._2(a2_plus_b_minus_2c, 2, 4)(3) == -1 # (2)^2 + 3 - 2*4
        assert bl._3(a2_plus_b_minus_2c, 2, 4)(3) == 2 # (2)^2 + 4 - 2*3

    def test_pipe(self):
        assert bl.pipe(4, add2, mul3) == 18

        assert [18, 14] == P(
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
        )

        assert bl.pipe(
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
                        add2,
                        [
                            _ + 1,
                            _ + 2
                        ]
                    )
                ]
            ]
        ) == [18, 18, 15, 16]

        assert bl.pipe(
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

        [a, b, c] = bl.pipe(
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
        y = bl.ref

        z = bl.pipe(
            self.x,
            { tf.name_scope('TEST'):
                (
                _ * 2,
                _ + 4,
                bl.store(y)
                )
            },
            _ ** 3
        )

        assert "TEST/" in y().name
        assert "TEST/" not in z.name

    def test_register_1(self):

        #register
        assert 5 == bl.pipe(
            3,
            bl.add(2)
        )

        #register_2
        assert 8 == bl.pipe(
            3,
            bl.pow(2)
        )

        #register_method
        assert "identity" == bl.get_function_name()

    def test_using_run(self):
        assert 8 == bl.using(3).add(2).add(3).run()

    def test_reference(self):
        add_ref = bl.ref

        assert 8 == bl.using(3).add(2).store(add_ref).add(3).run()
        assert 5 == add_ref()

    def test_ref_props(self):

        a = bl.ref
        b = bl.ref

        assert [7, 3, 5] == bl.pipe(
            1,
            add2, a.set,
            add2, b.set,
            add2,
            [
                bl.identity,
                a,
                b
            ]
        )

    def test_scope_property(self):

        assert "some random text" == bl.pipe(
            "some ",
            { DummyContext("random "):
            (
                lambda s: s + bl.Scope(),
                { DummyContext("text"):
                    lambda s: s + bl.Scope()
                }
            )
            }
        )

        assert bl.Scope() == None



#
# class TestBuilder(object):
#     """docstring for TestBuilder"""
#
#     x = tf.placeholder(tf.float32, shape=[None, 5])
#
#     def test_build(self):
#         bl = tb.build(self.x)
#         assert type(bl) == tb.Builder
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
