from tensorbuilder.builder import dsl
from fn import _
from tensorbuilder.builder import utils

class TestDSL(object):
    """docstring for TestDSL."""

    def test_compile(self):
        code = (_ + 1, _ * 2)
        f, refs = dsl.Compile(code, {})
        assert f(2) == 6

    def test_read(self):
        refs = dict(
            x=dsl.Ref(10)
        )
        code = ('x',)
        f, refs = dsl.Compile(code, refs)

        assert refs == refs #read doesnt modify
        assert f(None) == 10

    def test_write(self):
        code = (
            _ + 1, {'a'},
            _ * 2, {'b'},
            _ * 100, {'c'},
            ['c', 'a', 'b']
        )

        f, refs = dsl.Compile(code, {})

        assert [600, 3, 6] == f(2)

    def test_input(self):
        code = (
             {'a'},
            _ + 1,
            [
            (
                {(10,)},
                _ * 2
            )
            ,
                'a'
            ,
                utils.identity
            ]
        )

        f, refs = dsl.Compile(code, {})

        assert [20, 2, 3] == f(2)
