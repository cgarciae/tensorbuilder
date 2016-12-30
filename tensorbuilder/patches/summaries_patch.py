from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import inspect
import functools
from tensorbuilder import TensorBuilder
from phi import Builder, Obj

#########################
## Layer methods
#########################

class SummaryBuilder(Builder):
    pass

#Add property to TensorBuilder
TensorBuilder.summary = property(lambda self: SummaryBuilder(self._f))

#########################
## normal
#########################

SummaryBuilder.PatchAt(2, tf.summary,
    return_type_predicate = TensorBuilder,
    blacklist_predicate = Obj.isupper()
)

#########################
## make_*
#########################

def summary_wrapper(f):
    def summary_function(name, tensor, *args, **kwargs):

        f(name, tensor, *args, **kwargs)
        return tensor
    return summary_function

SummaryBuilder.PatchAt(2, tf.summary,
    return_type_predicate = TensorBuilder,
    method_wrapper = summary_wrapper,
    method_name_modifier = "create_{0}".format,
    blacklist_predicate = Obj.isupper(),
    explanation = """but the Tensor computed by {original_name} is not returned, instead its *2nd* argument (the numeric tensor) is returned untouched. This function is mostly used for side effects."""
)

