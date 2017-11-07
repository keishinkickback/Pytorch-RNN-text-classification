import torch
import torch.nn as nn

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

        # bn1
        # self.bn1 = nn.BatchNorm1d(embed_size)
        self.drop_en = nn.Dropout(p=0.8)

        # rnn module
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.8,
            batch_first=True,
            bidirectional=True
        )

        # self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.drop = nn.Dropout(p=0.8)
        self.fc = nn.Linear(hidden_size*2, num_output)

    def forward(self, x):
        '''
        :param x: (batch, time_step, input_size)
        :return: num_output size
        '''

        x = self.encoder(x)

        x = self.drop_en(x)

        # x = x.transpose(1, 2)
        # # print(x.size())
        # x = self.bn1(x)
        # x = x.transpose(1, 2)

        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state
        r_out, h_n = self.rnn(x, None)

        # only use the final output
        # out = r_out[:, -1, :]

        # use mean of outputs
        out = torch.mean(r_out, dim=1)

        # out = self.bn2(out)
        out = self.drop(out)
        out = self.fc(out)
        return out
