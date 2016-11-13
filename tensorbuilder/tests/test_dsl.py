from tensorbuilder.builder import dsl
from fn import _

class TestDSL(object):
    """docstring for TestDSL."""

    def test_compile(self):
        code = (_ + 1, _ * 2)
        f, refs = dsl.Compile(code, {})

        ast = dsl.parse(code)
        print(ast)

        f2, refs = ast.compile({})

        print(f2)

        assert f(2) == 6
