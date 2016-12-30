from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import tensorflow as tf
import inspect
from tensorbuilder import TensorBuilder
from phi import utils


f0_pred = (lambda x:
    "scope" in x or
    "device" in x
)

f2_pred = (lambda x:
    x in [
        "concat"
    ] or
    "_summary" in x
)

f1_blacklist = (lambda x:
    x in ["relu_layer", "device"] or
    x in TensorBuilder.__core__ or
    f0_pred(x) or
    f2_pred(x)
)

#tf
TensorBuilder.PatchAt(0, tf, whitelist_predicate=f0_pred)
TensorBuilder.PatchAt(1, tf, blacklist_predicate=f1_blacklist)
TensorBuilder.PatchAt(2, tf, whitelist_predicate=f2_pred)

#tf.nn
TensorBuilder.PatchAt(1, tf.nn, module_alias="tf.nn", blacklist_predicate=f1_blacklist)

# for name, f, module in f1s:
#     TensorBuilder.register_function_1(f, module)
#
# for name, f, module in f2s:
#     TensorBuilder.register_function_2(f, module)
