
from tensorbuilder.core import concrete_classes
from tensorbuilder import core

def class_patcher(patch):
    classes = concrete_classes.get(core.BuilderBase, core.BuilderTreeBase, core.ApplicativeBase)
    patch(*classes)
    return classes

def patched_tensorbuilder_classes():
    from tensorbuilder_patch import patch_classes
    return class_patcher(patch_classes)