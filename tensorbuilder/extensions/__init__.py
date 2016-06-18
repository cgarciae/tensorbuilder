
from classes import concrete_classes
from tensorbuilder import core

def class_patcher(patch):
    classes = concrete_classes(core.BuilderBase, core.BuilderTreeBase, core.ApplicativeBase)
    patch(*classes)
    return classes

def tensorbuilder_classes():
    from patches.tensorbuilder_patch import patch_classes
    return class_patcher(patch_classes)
