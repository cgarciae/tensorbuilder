from phi import utils
from tensorbuilder import TensorBuilder


@TensorBuilder.RegisterMethod("tb")
def rnn_placeholders_from_state(self, zero_state, name="rnn_state"):
    if isinstance(zero_state, tuple):
        return tuple([self.rnn_placeholders_from_state(substate, name=name) for substate in zero_state])
    else:
        return tf.placeholder(zero_state.dtype, shape=zero_state.get_shape(), name=name)

@TensorBuilder.RegisterMethod("tb")
def rnn_state_feed_dict(self, placeholders, values):
    return dict(zip(utils.flatten(placeholders), utils.flatten_list(values)))
