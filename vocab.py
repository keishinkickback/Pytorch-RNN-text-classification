from collections import defaultdict

import pandas as pd

import util as ut


class VocabBuilder():
    '''
    Read file and create word_to_index dictionary.
    This can truncate low-frequency words with min_sample option.
    '''
    def __init__(self, path_file=None, min_sample=1):
        # word count
        if path_file:
            word_count = self.count_from_file(path_file)
            # truncate low fq word
            self.word_to_index =\
                self.create_word_to_index(word_count, min_sample)
        else:
            raise RuntimeError('need path_file')


    def count_from_file(self, path_file):

        word_count = defaultdict(int)

        df = pd.read_csv(path_file,delimiter='\t')
        df['body'] = df['body'].apply(ut._tokenize)
        samples = df['body'].values.tolist()

        for sample in samples:
            for tkn in sample:
                word_count[tkn.lower()] += 1

        print('Original Vocab size:{}'.format(len(word_count)))

        return word_count

    def create_word_to_index(self,word_count, min_sample=1):

        # create vocab
        word_to_index = {}
        _removed = 0

        for tkn, fq in sorted(word_count.items()):
            if fq < min_sample:
                _removed += 1
            else:
                word_to_index[tkn] = len(word_to_index) + 2

        print('Turncated vocab size:{} (removed:{})'.\
            format(len(word_to_index),_removed))

        return word_to_index

    def get_word_index(self, inc_unknown=True, unknown_marker='__UNK__'):

        if inc_unknown and unknown_marker not in self.word_to_index:
            self.word_to_index[unknown_marker] = 1

        elif inc_unknown:
            raise RuntimeError('Unknown Marker:{} is already used.'.\
                               format(unknown_marker))

        self.word_to_index['__PADDING__'] = 0

        return self.word_to_index


if __name__ == "__main__":

    v_builder = VocabBuilder(path_file='data/data_1_train.tsv', min_sample=10)
    d = v_builder.get_word_index()
    print (d['__UNK__'])
    for k, v in sorted(d.items())[:100]:
        print (k,v)

