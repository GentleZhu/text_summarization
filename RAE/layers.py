import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import cPickle
import utils

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