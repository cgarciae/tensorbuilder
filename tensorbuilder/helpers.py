from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import operator

def count_parameters(graph):
    import operator
    with graph.as_default():
        prod = lambda iterable: reduce(operator.mul, iterable, 1)
        dims = [ list([ int(d) for d in  v.get_shape()]) for v in tf.all_variables() ]
        vars = [ prod(shape) for shape in dims ]
        return sum(vars)