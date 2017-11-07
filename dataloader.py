import torch
import pandas as pd
import numpy as np

import util as ut

class TextClassDataLoader(object):

    def __init__(self, path_file, word_to_index, batch_size=32):

        self.batch_size = batch_size
        self.word_to_index = word_to_index

        df = pd.read_csv(path_file,delimiter='\t')
        df['body'] = df['body'].apply(ut._tokenize)
        df['body'] = df['body'].apply(self.generate_indexifyer())
        self.samples = df.values.tolist()

        self.shuffle_indices()
        self.n_batches = int(len(self.samples) / self.batch_size)
        self.max_length = self.get_max_length()
        self.report()

    def shuffle_indices(self):
        self.indices = np.random.permutation(len(self.samples))
        self.index = 0
        self.batch_index = 0

    def get_max_length(self):
        length = 0
        for sample in self.samples:
            length = max(length, len(sample[1]))
        return length

    def generate_indexifyer(self):

        def indexify(lst_text):
            indices = []
            for word in lst_text:
                if word in self.word_to_index:
                    indices.append(self.word_to_index[word])
                else:
                    indices.append(self.word_to_index['__UNK__'])
            return indices

        return indexify

    @staticmethod
    def _padding(batch):
        batch = sorted(batch, key=lambda x: len(x[1]))
        size = len(batch[-1][1])
        for i, x in enumerate(batch):
            missing = size - len(x[1])
            batch[i][1] = [0 for _ in range(missing)] + batch[i][1]
        return batch

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1
        batch = self._padding(batch)
        label, string = tuple(zip(*batch))

        label = torch.LongTensor(label)
        string = torch.LongTensor(string)
        return string, label

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print sample

    def report(self):
        print '# samples: {}'.format(len(self.samples))
        print 'max len: {}'.format(self.max_length)
        print '# vocab: {}'.format(len(self.word_to_index))
        print '# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size)
