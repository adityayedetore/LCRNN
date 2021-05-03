import torch.nn as nn
from torch.autograd import Variable
from locallyconnectedMLP import LocallyConnectedMLP

import customLSTM


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, lcrnn="none", lc_kernel_size=100, lc_n_layers=10, lc_activation="Sigmoid", lc_conv=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['customLSTM']:
            self.rnn = customLSTM.LSTMModel(
                ninp, nhid)  # , nlayers, dropout=False)
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
        if lcrnn in ['none', 'inside', 'outside', 'both']:
            if lcrnn == 'inside' or lcrnn == 'both':
                activation = getattr(nn, lc_activation)()
                self.LCLs_inside = LocallyConnectedMLP(n_layers=lc_n_layers,
                                                                     activation_fn=activation,
                                                                     input_dim=[
                                                                         nhid] * lc_n_layers,
                                                                     output_dim=[
                                                                         nhid] * lc_n_layers,
                                                                     kernel_size=[
                                                                         lc_kernel_size] * lc_n_layers,
                                                                     stride=[1] * lc_n_layers,
                                                                     conv=lc_conv)
            if lcrnn == 'outside' or lcrnn == 'both':
                activation = getattr(nn, lc_activation)()
                self.LCLs_outside = LocallyConnectedMLP(n_layers=lc_n_layers,
                                                                     activation_fn=activation,
                                                                     input_dim=[
                                                                         nhid] * lc_n_layers,
                                                                     output_dim=[
                                                                         nhid] * lc_n_layers,
                                                                     kernel_size=[
                                                                         lc_kernel_size] * lc_n_layers,
                                                                     stride=[1] * lc_n_layers,
                                                                     conv=lc_conv)
        else:
            raise ValueError("""An invalid option for `--lcrnn` was supplied,
                             options are ['none', 'inside', 'outside' or 'both']""")


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
        # The locally connected weights are initialized in the LocallyConnectedLayer1D class

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        if self.lcrnn == 'inside' or self.lcrnn == 'both': 
            hidden = self.LCLs_inside(hidden)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        if self.lcrnn == 'outside' or self.lcrnn == 'both': 
            output = self.LCLs_outside(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
