import tensorflow as tf
from tensorbuilder.core import utils

builders_blacklist = []


def rnn_placeholders_from_state(applicative, zero_state, name="rnn_state"):
    if isinstance(zero_state, tuple):
        return tuple([applicative.rnn_placeholders_from_state(substate, name=name) for substate in zero_state])
    else:
        return tf.placeholder(zero_state.dtype, shape=zero_state.get_shape(), name=name)

def rnn_state_feed_dict(applicative, placeholders, values):
    return dict(zip(utils.flatten(placeholders), utils.flatten(values)))

def patch_classes(Builder, BuilderTree, Applicative):
    Applicative.register_method(rnn_placeholders_from_state, "tensorbuilder.Applicative")
    Applicative.register_method(rnn_state_feed_dict, "tensorbuilder.Applicative")
