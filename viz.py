from graphviz import Digraph
import torch
from torch.autograd import Variable
from torchviz import make_dot

# model

grph = make_dot(self.model)
