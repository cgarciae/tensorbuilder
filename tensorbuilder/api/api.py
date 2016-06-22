import builder
import builder_tree
import applicative
from tensorbuilder import tensordata

class API(applicative.Applicative):
    """docstring for API"""
    def __init__(self, f):
        super(API, self).__init__(f)

    def build(self, tensor):
        return self.Builder(tensor)

    def branches(self, builder_iterable):
        return self.BuilderTree(builder_iterable)

API.data = tensordata.data

API.Builder = builder.Builder
API.BuilderTree = builder_tree.BuilderTree
API.Applicative = applicative.Applicative
