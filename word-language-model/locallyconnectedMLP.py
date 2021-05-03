import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from collections import OrderedDict

class LocallyConnectedMLP(nn.Module):
    def __init__(self, 
                 n_layers: int, 
                 activation_fn: nn.Module,
                 input_dim: List[int], 
                 output_dim: List[int], 
                 kernel_size: List[int], 
                 stride: List[int], 
                 padding: bool = True, 
                 bias: bool = True):
      """
      Creates multiple locally connected layers. For the arguments defining
      locally connected layers, need to pass in a list of integers, one for each
      layer.
      Locally connected layers are highly sensitive to gradient flow. Need to be
      careful about not making kernel size too small or too large, and need to 
      visualize what the network will look like, making sure that stride doesn't
      make most of the nodes in the layer not connected to anything.

      Example call:
      model = LocallyConnectedMLP(n_layers = 4, 
                                  activation_fn = nn.Sigmoid(), 
                                  input_dim = [10,10,10,10], 
                                  output_dim = [10,10,10,1], 
                                  kernel_size = [2,2,2,10], 
                                  stride = [1,1,1,1])
      """
      super(LocallyConnectedMLP, self).__init__()

      lcstack = []
      lcstack.append(('first', nn.Linear(2,10)))
      for j in range(n_layers-1):
        args = []
        for arg in [input_dim, output_dim, kernel_size, stride]:
            args.append(arg[j])
        lc = LocallyConnectedLayer1d(args[0], args[1], args[2], args[3], padding, bias)
        layer_name = 'locallyconnected' + str(j+1)
        lcstack.append((layer_name,lc))
        activation_name = 'activation' + str(j+1)
        lcstack.append((activation_name, activation_fn))
      # final layer
      lc = LocallyConnectedLayer1d(input_dim[j+1], output_dim[j+1], kernel_size[j+1], stride[j+1], padding, bias)
      layer_name = 'locallyconnected' + str(j+2)
      lcstack.append((layer_name,lc))

      self.lcmlp = nn.Sequential(OrderedDict(lcstack))

    def forward(self, x):
      out = self.lcmlp(x)
      return out
