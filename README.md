# RNN-based short text classification

- This is for multi-class short text classification.
- Model is built with Word Embedding, LSTM ( or GRU), and Fully-connected layer by [Pytorch](http://pytorch.org).
- A mini-batch is created by 0 padding and processed by using torch.nn.utils.rnn.PackedSequence.
- Cross-entropy Loss + Adam optimizer.
- Support pretrained word embedding ([GloVe](https://nlp.stanford.edu/projects/glove/)).
## Model
- Embedding --> Dropout --> LSTM(GRU) --> Dropout --> FC.



## Preprocessing
- The following command will download the dataset used in
 [Learning to Classify Short and Sparse Text & Web with Hidden Topics from Large-scale Data Collections](http://wwwconference.org/www2008/papers/pdf/p91-phanA.pdf)
 from [here](http://jwebpro.sourceforge.net/data-web-snippets.tar.gz) and process it for training.
- Also it download GloVe embeddings.
```
python preprocess.py
```

## Training

- The following command starts training. Run it with ```-h``` for optional arguments.

```
python main.py
```