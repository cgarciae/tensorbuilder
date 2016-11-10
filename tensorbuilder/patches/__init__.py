#import layers_patch
import tensorflow_patch
import summaries_patch
import layers_patch

def patch(TensorBuilder):
    tensorflow_patch.patch(TensorBuilder)
    summaries_patch.patch(TensorBuilder)
    layers_patch.patch(TensorBuilder)
