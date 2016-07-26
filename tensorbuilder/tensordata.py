import asq
from asq.initiators import query
import asq.queryables
import random
from itertools import islice, izip_longest
import numpy as np
from core.utils import immutable
import tensorflow as tf

"""
"""
def data(*args, **kwargs):
    return Data(*args, **kwargs)

class Data(object):
    """docstring for Data"""
    def __init__(self, _iterator=None, **sources):
        super(Data, self).__init__()
        self.sources = sources
        self.__dict__.update(sources)

        self._iterator = _iterator if _iterator else lambda: self._raw_data()
        self.batch = None
        self.patch = None


    def copy(self):
        return Data(_iterator=self._iterator, **self.sources)

    def __iter__(self):
        return self._iterator()

    def enumerated(self):
        return enumerate(self._iterator())


    def split(self, *splits):
        """docstring for Batcher"""

        data_length = len(self.x)

        indexes = range(data_length)
        random.shuffle(indexes)

        splits = [0] + list(splits)
        splits_total = sum(splits)

        return (
            query(splits)
            .scan()
            .select(lambda n: int(data_length * n / splits_total))
            .then(_window, n=2)
            .select(lambda (start, end): np.array(indexes[start:end]))
            .select(lambda split: Data(**{k: source[split,:] for (k, source) in self.sources.iteritems()}))
            .to_list()
        )


    @immutable
    def raw_data(self):
        self._iterator = lambda: self._raw_data()
        return self

    def _raw_data(self):
        yield self


    @immutable
    def batches_of(self, batch_size):
        """
        docstring for Batcher
        """
        _iterator = self._iterator
        self._iterator = lambda: self._batch(batch_size, _iterator)
        return self

    def _batch(self, batch_size, _iterator):
        for data in _iterator():
            length = len(data.x)
            sample = np.random.choice(length, batch_size)
            new_data = Data(**{k: source[sample] for (k, source) in data.sources.iteritems()})

            yield new_data

    @immutable
    def epochs(self, epochs):
        """docstring for Batcher"""
        _iterator = self._iterator
        self._iterator = lambda: self._epochs(epochs, _iterator)
        return self

    def _epochs(self, epochs, _iterator):
        for epoch in range(epochs):
            for data in _iterator():
                data.epoch = epoch
                yield data


    def placeholders(self, *args):
        return list(self._placeholders(*args))

    def _placeholders(self, *args):
         for source_name in args:
             source = self.sources[source_name]
             shape = [None] + list(source.shape)[1:]
             yield tf.placeholder(tf.float32, shape=shape)


    def run(self, sess, tensor, tensors={}, **feed):

        try:
            tensor = tensor.tensor()
        except:
            pass

        feed = { feed[k]: self.sources[k] for k in feed }
        feed.update(tensors)

        return sess.run(tensor, feed_dict=feed)


def _window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def _then(q, fn, *args, **kwargs):
    return query(fn(q, *args, **kwargs))

asq.queryables.Queryable.then = _then

if __name__ == '__main__':
    x = np.array(range(1200)).reshape(400, 3)
    y = np.array(range(400)).reshape(400, 1)

    d = data(x=x, y=y)
    [training, validation, test] = d.split(0.6, 0.2, 0.2)
    print([training, validation, test])

    for dat in training.batches(4).epochs(10):
        print(dat.x, dat.batch, dat.epoch)
