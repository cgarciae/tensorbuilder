import tensorflow as tf
import inspect
from tensorbuilder import TensorBuilder
from phi import utils, patch


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
patch.builder_with_members_from_0(TensorBuilder, tf, whitelist=f0_pred)
patch.builder_with_members_from_1(TensorBuilder, tf, blacklist=f1_blacklist)
patch.builder_with_members_from_2(TensorBuilder, tf, whitelist=f2_pred)

#tf.nn
patch.builder_with_members_from_1(TensorBuilder, tf.nn, module_alias="tf.nn", blacklist=f1_blacklist)

# for name, f, module in f1s:
#     TensorBuilder.register_function_1(f, module)
#
# for name, f, module in f2s:
#     TensorBuilder.register_function_2(f, module)
