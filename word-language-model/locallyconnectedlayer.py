import torch
import torch.nn as nn
import torch.nn.functional as F

class LocallyConnectedLayer1d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=True, bias=False):
      """
      Defines one locally connected layer for one dimensional vector input.
      NOTE: This model only takes one-dimensional inputs. Batched inputs are also fine.

      input_dim: column size of weight matrix
      output_dim: row size of weight matrix
      kernel_size: number of local connections per parameter
      stride: number of strides of local connections, CANNOT BE ZERO
      padding: whether or not to zero pad
      bias: whether or not to have a bias term
      """
      super(LocallyConnectedLayer1d, self).__init__()
      # Goal is to create sparse locally connected weight matrix size output_dim x input_dim
      # Instead of doing that, dense matrix of size output_dim x kernel_size is equivalent
      # Given special operation substitute for matrix multiplication
      self.weight = nn.Parameter(
          torch.randn(output_dim, kernel_size)
      )
      if bias:
          self.bias = nn.Parameter(
              torch.randn(output_dim)
          )
      else:
          self.register_parameter('bias', None)
      self.kernel_size = kernel_size
      self.stride = stride

      if padding == True:
        # Compute the amount needed to pad input vector with 0s by
        # In order to make the unfolding operation result in row dim same as output_dim
        pad_size = (stride * (output_dim - 1)) + kernel_size - input_dim
        self.input_dim = input_dim + pad_size
        self.pad = True
      else:
        # Alternative to padding is to repeat last row of unfolded input matrix
        # Until number of rows is the same as output_dim
        resulting_dim = ((input_dim - kernel_size) / stride) + 1
        self.extra_dim = output_dim - resulting_dim
        self.pad = False

    def forward(self, x):
      k = self.kernel_size
      s = self.stride

      instance_dim = len(x.size())-1

      # Executes padding strategy to resize input vector x for matrix multiplication shortcut
      if self.pad:
        pad_size = self.input_dim - x.size()[instance_dim]
        # If pad size is positive need to pad, else need to remove data
        if pad_size >= 0:
          # How much to pad each dimension
          pad = (0, pad_size) #TODO this might be wrong, need to check this works as intended
          x = F.pad(x, pad=pad)
          # Unfold x by striding over the input features by kernel size and stride
          x = x.unfold(instance_dim, k, s)
        else: # If pad size is negative, need to remove rows from unfolded x
          x = x.unfold(instance_dim, k, s)
          if instance_dim == 0:
            x = x[:pad_size]
          else:
            x = x[:,:pad_size,:]
      else: # If not padding, need to duplicate last unfolded row, this is the uglier method
        x = x.unfold(instance_dim, k, s)
        for i in self.extra_dim:
          if instance_dim == 0:
            x = torch.cat((x, x[-1]), dim=instance_dim)
          else:
            x = torch.cat((x, x[:,-1,:]), dim=instance_dim)

      # Do elementwise multiplication of x and weights and then sum across the rows
      # NOTE: this is the same as doing matrix multiplication with sparse weight matrix, but faster
      out = torch.sum(x * self.weight, dim=instance_dim+1)
      if self.bias is not None:
          out += self.bias
      return out
