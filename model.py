import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, hidden_size=64,
                                          num_layers=1,batch_first=True):

        '''
        :param vocab_size: vocab size
        :param embed_size: embedding size
        :param num_output: number of output (classes)
        :param hidden_size: hidden size of rnn module
        :param num_layers: number of layers in rnn module
        :param batch_first: batch first option
        '''

        super(RNN, self).__init__()

        # embedding
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.drop_en = nn.Dropout(p=0.8)

        # rnn module
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5,
            batch_first=True,
            bidirectional=True
        )

        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, num_output)

    def forward(self, x, seq_lengths):
        '''
        :param x: (batch, time_step, input_size)
        :return: num_output size
        '''

        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)
        packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(),batch_first=True)

        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state
        packed_output, (ht, ct) = self.rnn(packed_input, None)

        # use mean of outputs
        out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)

        row_indices = torch.arange(0, x.size(0)).long()
        col_indices = seq_lengths - 1
        if next(self.parameters()).is_cuda:
            row_indices = row_indices.cuda()
            col_indices = col_indices.cuda()

        last_tensor=out_rnn[row_indices, col_indices, :]
        # fc_input = torch.mean(last_tensor, dim=1)

        fc_input = self.bn2(last_tensor)
        out = self.fc(fc_input)
        return out
