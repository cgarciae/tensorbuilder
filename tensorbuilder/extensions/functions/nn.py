import tensorflow as tf

def polynomic(name=None):
	def _polynomic(tensor):
	    size = m = int(tensor.get_shape()[1])
	    pows = map(lambda n: tf.pow(tensor[:, n], n + 1), range(size))
    	return tf.transpose(tf.pack(pows), name=name)
  	return polynomic
