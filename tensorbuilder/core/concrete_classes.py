

def get(BuilderBase, BuilderTreeBase, ApplicativeBase):
    class Builder(BuilderBase):
        """docstring for Builder"""
        def __init__(self, tensor):
            super(Builder, self).__init__(tensor)

        def BuilderTree(self, builder_iterable):
            return BuilderTree(builder_iterable)

    class BuilderTree(BuilderTreeBase):
        """docstring for BuilderTree"""
        def __init__(self, builder_iterable):
            super(BuilderTree, self).__init__(builder_iterable)

        def Builder(self, tensor):
            return Builder(tensor)

    class Applicative(ApplicativeBase):
        """docstring for Applicative"""
        def __init__(self, f):
            super(Applicative, self).__init__(f)

        def Builder(self, tensor):
            return Builder(tensor)

    return (Builder, BuilderTree, Applicative)
