from tensorbuilder import tensorbuilder as tb
import tensorflow as tf

class TestTensorBuilder(object):
    """docstring for TestBuilder"""

    @classmethod
    def setup_method(self):
        pass

    def test_patch(self):
        assert tb.matmul
