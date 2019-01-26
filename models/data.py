import torch
from torch.utils import data

#TODO: move negative samples from model to here
class SummDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.list_IDs[index,:]
        y = self.labels[index]

        return X, y


class KnowledgeEmbedDataset(data.Dataset):
  'Characterizes dataset for KnowledgeEmb'
  def __init__(self, data):
        'Initialization'
        self.edges = data

  def __len__(self):
        'Denotes the total number of samples'
        return max(len(d) for d in self.edges)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        return tuple(d[index % len(d)] for d in self.edges)