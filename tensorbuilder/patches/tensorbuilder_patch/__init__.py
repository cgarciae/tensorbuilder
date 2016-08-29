import tensorflow as tf
from tensorbuilder.core.builders import BuilderBase, BuilderTreeBase
from tensorbuilder.core.applicative import ApplicativeBase
from tensorbuilder.core import utils
import numpy as np
import tflearn as tl
import inspect
import summaries
import tensorflow_ops
import tflearn_ops
import scope_ops
import applicative_ops
import custom_layer_ops
import builder_custom_ops
import api_functions_patch

def patch_classes(Builder, BuilderTree, Applicative):

    builders_blacklist = (
        summaries.builders_blacklist +
        builder_custom_ops.builders_blacklist +
        custom_layer_ops.builders_blacklist +
        scope_ops.builders_blacklist +
        api_functions_patch.builders_blacklist
    )
    applicative_builder_blacklist = (
        ["copy", "compose"] +
        scope_ops.builders_blacklist +
        ApplicativeBase.__core__ +
        [ "with_" + v for v in scope_ops.builders_blacklist ]
    )
    applicative_tree_blacklist = (
        ["copy", "connect_layer", "compose"] +
        scope_ops.builders_blacklist +
        ApplicativeBase.__core__ +
        [ "with_" + v for v in scope_ops.builders_blacklist ]
    )

    #######################
    ### patch_classes
    #######################

    builder_custom_ops.patch_classes(Builder, BuilderTree, Applicative)
    custom_layer_ops.patch_classes(Builder, BuilderTree, Applicative, builders_blacklist)
    scope_ops.patch_classes(Builder, BuilderTree, Applicative)
    summaries.patch_classes(Builder, BuilderTree, Applicative)
    tensorflow_ops.patch_classes(Builder, BuilderTree, Applicative)
    tflearn_ops.patch_classes(Builder, BuilderTree, Applicative)
    api_functions_patch.patch_classes(Builder, BuilderTree, Applicative)


    #Applicative should go last
    applicative_ops.patch_classes(
        Builder, BuilderTree, Applicative,
        applicative_builder_blacklist, applicative_tree_blacklist
    )

