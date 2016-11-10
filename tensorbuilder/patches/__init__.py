#import layers_patch
import tensorflow_patch
import summaries_patch

def patch(TensorBuilder):
    tensorflow_patch.patch(TensorBuilder)
    summaries_patch.patch(TensorBuilder)
