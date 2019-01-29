import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import utils
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(t.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(t.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = t.mm(input, self.weight)
        output = t.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DistMult(nn.Module):
    def __init__(self, num_classes, input_features):
        super(DistMult, self).__init__()
        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        self.weight = nn.Parameter(t.randn(num_classes, input_features), requires_grad=True)

    def forward(self, idx, input):
        # See the autograd section for explanation of what happens here.
        return input * self.weight[idx, :]

class BiLinear(nn.Module):
    def __init__(self, num_classes, input_features):
        super(DistMult, self).__init__()
        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        self.weight = nn.Parameter(t.randn(num_classes, input_features, input_features), requires_grad=True)

    def forward(self, idx, input_x, input_y):
        # See the autograd section for explanation of what happens here.
        return input_x * self.weight[idx, :]


class KnowledgeAwareAttention(nn.Module):
    """
    A knowledge aware attention layer where the attention weight is calculated as
    a = x R q
    where x is the input, q is the query, and f is additional position features.
    """
    
    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.rlinear = nn.Embedding(self.num_classes + 2, self.embed_size, sparse=False)

        self.init_weights()

    def init_weights(self):
        self.rlinear.weight.data.normal_(std=0.001)
        self.rlinear[0].weight.data.zero_()
        self.rlinear[0].requires_grad = False
    
    def forward(self, x, x_mask, q, f):
        """
        x : batch_size * input_size
        q : batch_size * batch_size, with relation type
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()
        r_input = self.rlinear(Variable(q.cuda()))

        attn = (r_input * x.unsqueeze(0).repeat(batch_size,1,1) * x.unsqueeze(1).repeat(1,batch_size,1)).sum(2)
        weights = F.softmax(attn)
        outputs = weights.matmul(x)
        return outputs