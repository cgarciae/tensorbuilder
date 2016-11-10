#import layers_patch
import tensorflow_patch

def patch(TensorBuilder):
    tensorflow_patch.patch(TensorBuilder)
