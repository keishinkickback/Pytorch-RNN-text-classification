import os
import subprocess
from util import create_tsv_file

if __name__ == '__main__':

    subprocess.call(['bash', '-c', 'wget http://jwebpro.sourceforge.net/data-web-snippets.tar.gz'])
    subprocess.call(['bash', '-c', 'tar xvzf data-web-snippets.tar.gz'])

    subprocess.call(['bash','-c', 'wget http://nlp.stanford.edu/data/glove.6B.zip'])
    subprocess.call(['bash', '-c', 'unzip glove.6B.zip'])

    if not os.path.exists('data'):
        os.mkdir('data')

    if not os.path.exists('glove'):
        os.mkdir('glove')

    subprocess.call(['bash', '-c', 'mv glove.6B.*d.txt glove'])
    create_tsv_file('data-web-snippets/train.txt', 'data/train.tsv')
    create_tsv_file('data-web-snippets/test.txt', 'data/test.tsv')
