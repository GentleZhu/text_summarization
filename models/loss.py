import torch
import torch.nn as nn
import torch.nn.functional as F


class NCE_SIGMOID(nn.Module):
    """Negative sampling loss as proposed by T. Mikolov et al. in Distributed
    Representations of Words and Phrases and their Compositionality.
    """
    def __init__(self):
        super(NCE_SIGMOID, self).__init__()
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, scores):
        """Computes the value of the loss function.
        Parameters
        ----------
        scores: autograd.Variable of size (batch_size, num_noise_words + 1)
            Sparse unnormalized log probabilities. The first element in each
            row is the ground truth score (i.e. the target), other elements
            are scores of samples from the noise distribution.
        """
        k = scores.size()[1] - 1
        return -torch.sum(
            self._log_sigmoid(scores[:, 0])
            + torch.sum(self._log_sigmoid(-scores[:, 1:]), dim=1) / k
        ) / scores.size()[0]

class NCE_HINGE(nn.Module):
    """docstring for NCE_HINGE"""
    def __init__(self):
        super(NCE_HINGE, self).__init__()

    def forward(self, scores):
        return torch.sum(F.relu(scores[:, 1:] - scores[:, 0].unsqueeze(1) + 1)) / scores.size()[0]
        