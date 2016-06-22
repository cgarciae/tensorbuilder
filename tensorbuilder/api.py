import extensions
import tensordata

Builder, BuilderTree, Applicative = extensions.patched_tensorbuilder_classes()

__builder__ = Builder(None)
__tree__ = BuilderTree([])
__applicative__ = Applicative(lambda x: x)

build = __builder__.build
branches = __tree__.branches
pipe = __applicative__.pipe
data = tensordata.data
dl = __applicative__

__all__ = ["Builder", "BuilderTree", "Applicative", "build", "pipe", "data", "dl"]
