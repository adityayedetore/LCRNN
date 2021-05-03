import torch.nn as nn
from torch.autograd import Variable
from locallyconnectedMLP import LocallyConnectedMLP

import customLSTM


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, lcrnn=False, lc_kernel_size=100, lc_n_layers=10, lc_activation="Sigmoid"):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['customLSTM']:
            self.rnn = customLSTM.LSTMModel(
                ninp, nhid)  # , nlayers, nhid, bias=True)

            # input_dim, hidden_dim, layer_dim, output_dim, bias=True):

        elif rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh',
                                'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers,
                              nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.lcrnn = lcrnn
        if lcrnn:
            activation = getattr(nn, lc_activation)()
            self.LCLs_after_recurrent_unit = LocallyConnectedMLP(n_layers=lc_n_layers,
                                                                 activation_fn=activation,
                                                                 input_dim=[
                                                                     nhid] * lc_n_layers,
                                                                 output_dim=[
                                                                     nhid] * lc_n_layers,
                                                                 kernel_size=[
                                                                     lc_kernel_size] * lc_n_layers,
                                                                 stride=[1] * lc_n_layers)
        else:
            self.LCLs_after_recurrent_unit = nn.Identity()

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        # The weights are initialized properly in the LocallyConnectedLayer1D class

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output = self.LCLs_after_recurrent_unit(output)
        decoded = self.decoder(output.view(
            output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
