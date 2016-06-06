import tensorflow as tf
import inspect


def polynomic(name=None):
	def _polynomic(tensor):
	    size = m = int(tensor.get_shape()[1])
	    pows = map(lambda n: tf.pow(tensor[:, n], n + 1), range(size))
    	return tf.transpose(tf.pack(pows), name=name)
  	return polynomic



# Dynamically generate functions with the same name as in the "tf.nn" but generate partials that 
# expect the tensor and call the original function from "tf.nn". This is done soy they can easily be used
# with the builder API, avoids the need of creating lambdas
# for (name, fn) in inspect.getmembers(tf.nn, inspect.isfunction):
#  		exec("""
 			
# def {0}(*args, **kwargs):

# 	def {0}_partial(tensor):
# 		return tf.nn.{0}(tensor, *args, **kwargs)

# 	return {0}_partial

# 			""".format(name))

