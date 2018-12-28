import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import loss as L
from layers import DistMult

class KnowledgeD2V(nn.Module):
    def __init__(self, num_words, num_docs, embed_size, kb_emb_size, weights=None, relational_bias=None):
        """
        :param num_classes: An int. The number of possible classes.
        :param embed_size: An int. EmbeddingLockup size
        :param num_sampled: An int. The number of sampled from noise examples
        :param weights: A list of non negative floats. Class weights. None if
            using uniform sampling. The weights are calculated prior to
            estimation and can be of any form, e.g equation (5) in [1]
        """

        super(KnowledgeD2V, self).__init__()

        self.num_words = num_words
        self.num_docs = num_docs
        self.embed_size = embed_size
        self.kb_emb_size = kb_emb_size

        self.word_embed = nn.Embedding(self.num_words, self.embed_size, sparse=True)
        self.word_embed.weight = Parameter(t.FloatTensor(self.num_words, self.embed_size).uniform_(-1, 1))
        
        self.out_embed = nn.Embedding(self.num_words, self.embed_size, sparse=True)
        self.out_embed.weight = Parameter(t.FloatTensor(self.num_words, self.embed_size).uniform_(-1, 1))

        self.doc_embed = nn.Embedding(self.num_docs, self.embed_size, sparse=True)
        self.doc_embed.weight = Parameter(t.FloatTensor(self.num_docs, self.embed_size).uniform_(-1, 1))
        self.weights = weights
        if self.weights is not None:
            assert min(self.weights) >= 0, "Each weight should be >= 0"

            self.weights = Variable(t.from_numpy(weights)).float()

        if relational_bias is not None:
            self.relational_bias = True
            self.r_linear = DistMult(len(relational_bias), self.kb_emb_size)
            self.W_r = nn.Linear(self.embed_size, self.kb_emb_size)
            self.W_r.weight.data.normal_()
            #self.W_r.weight.data.normal_(std=0.001)
        else:
            self.relational_bias = False

        self.nce_loss = L.NCE_SIGMOID()
        self.hinge_loss = L.NCE_HINGE()

    def sample(self, num_sample):
        """
        draws a sample from classes based on weights
        """
        return t.multinomial(self.weights, num_sample, True)

    def forward(self, input_labels, out_labels, num_sampled):
        """
        """
        [batch_size, window_size] = input_labels.size()
        out_labels.squeeze_()
        use_cuda = self.out_embed.weight.is_cuda
        #print(input_labels[:, -1], self.num_docs)
        relational_mask = t.ge(input_labels[:, -1], self.num_docs)
        context_idx = t.nonzero(1 - relational_mask).squeeze().view(-1)
        sub_idx = t.nonzero(relational_mask).squeeze().view(-1)

        doc_ids = t.masked_select(input_labels[:, -1], 1 - relational_mask)
        
        # bug
        # num_doc x 
        context_ids = input_labels[:, :-1].index_select(0, context_idx)

        if self.weights is not None:
            noise_sample_count = batch_size * num_sampled
            draw = self.sample(noise_sample_count)
            noise = draw.view(batch_size, num_sampled)
        else:
            noise = Variable(t.Tensor(batch_size, num_sampled).
                             uniform_(0, self.num_words - 1).long())
        #print(t.masked_select(out_labels[:,0], 1 - relational_mask).shape)
        target_labels = t.masked_select(out_labels, 1 - relational_mask).unsqueeze(1)
        target_noise_ids = t.cat((target_labels, noise[:doc_ids.size()[0], :]), dim=1)
        
        if use_cuda:
            doc_ids = doc_ids.cuda()
            context_ids = context_ids.cuda()
            target_noise_ids = target_noise_ids.cuda()
        # relational bias loss and DM loss

        # concatenate version
        # x = t.cat(self.doc_embed[doc_ids, :], torch.sum(self.word_embed[context_ids, :], dim=1))
        # average version
        x = t.add(
            self.doc_embed(doc_ids), t.sum(self.word_embed(context_ids), dim=1))

        output = t.bmm(
            x.unsqueeze(1),
            self.out_embed(target_noise_ids).permute(0, 2, 1)).squeeze()

        if self.relational_bias:
            if relational_mask.sum() == 0:
                return self.nce_loss(output), 0.0
            sub_ids = input_labels[:, 0].index_select(0, sub_idx).cuda()
            r_types = t.masked_select(input_labels[:, -1], relational_mask).cuda() - (self.num_docs+1)
            obj_ids = out_labels.masked_select(relational_mask).unsqueeze(1)
            relation_noise_ids = t.cat((obj_ids, noise[doc_ids.size()[0]:, :]), dim=1).cuda()
            x_proj = self.W_r(self.word_embed(sub_ids)).view(-1, self.kb_emb_size)
            y_proj = self.W_r(self.word_embed(relation_noise_ids)).view(-1, num_sampled + 1, self.kb_emb_size)
            r_x = self.r_linear(r_types, x_proj)
            r_output = t.bmm(
                r_x.unsqueeze(1),
                y_proj.permute(0, 2, 1)).squeeze(dim=1)
            #debug
            #assert r_output.size()[1] == 6
            #return self.nce_loss(r_output)
            
            return self.nce_loss(r_output) + self.nce_loss(output), self.nce_loss(r_output).float()
        
        return self.nce_loss(output), 0.0

    def input_embeddings(self):
        return self.word_embed.weight.data.cpu().numpy()

    def doc_embeddings(self):
        return self.doc_embed.weight.data.cpu().numpy()

# PTE is only applicable to one-layer label, we can run that
# Document representations-two ways: graph-way(use document to predict word), (DM)
# Start with fewer relationships, still find important asepcts(using attention), I will do this model
# ask them to read those papers: PTE/DOC2CUBE/HEER/Paragraph2Vec/draft, discuss tomorrow 8AM
# doing a heuristic linker
class KnowledgeAttnD2V(nn.Module):
    def __init__(self, num_words, num_docs, embed_size, kb_emb_size, weights=None, relational_bias=None):
        """
        :param num_classes: An int. The number of possible classes.
        :param embed_size: An int. EmbeddingLockup size
        :param num_sampled: An int. The number of sampled from noise examples
        :param weights: A list of non negative floats. Class weights. None if
            using uniform sampling. The weights are calculated prior to
            estimation and can be of any form, e.g equation (5) in [1]
        """

        super(KnowledgeD2V, self).__init__()

        self.num_words = num_words
        self.num_docs = num_docs
        self.embed_size = embed_size
        self.kb_emb_size = kb_emb_size

        self.word_embed = nn.Embedding(self.num_words, self.embed_size, sparse=True)
        self.word_embed.weight = Parameter(t.FloatTensor(self.num_words, self.embed_size).uniform_(-1, 1))
        
        self.out_embed = nn.Embedding(self.num_words, self.embed_size, sparse=True)
        self.out_embed.weight = Parameter(t.FloatTensor(self.num_words, self.embed_size).uniform_(-1, 1))

        self.doc_embed = nn.Embedding(self.num_docs, self.embed_size, sparse=True)
        self.doc_embed.weight = Parameter(t.FloatTensor(self.num_docs, self.embed_size).uniform_(-1, 1))
        self.weights = weights
        if self.weights is not None:
            assert min(self.weights) >= 0, "Each weight should be >= 0"

            self.weights = Variable(t.from_numpy(weights)).float()

        if relational_bias is not None:
            self.relational_bias = True
            self.r_linear = DistMult(len(relational_bias), self.kb_emb_size)
            self.W_r = nn.Linear(self.embed_size, self.kb_emb_size)
            self.W_r.weight.data.normal_()
            #self.W_r.weight.data.normal_(std=0.001)
        else:
            self.relational_bias = False

        self.nce_loss = L.NCE_SIGMOID()
        self.hinge_loss = L.NCE_HINGE()

    def sample(self, num_sample):
        """
        draws a sample from classes based on weights
        """
        return t.multinomial(self.weights, num_sample, True)

    def forward(self, input_labels, out_labels, num_sampled):
        """
        Design 1:
            input_labels: batch_size * (num_context + word_predict)
            kb_inputs: batch_size * num_context * num_neighbors(word_id
            kb_types: batch_size * num_context * num_neighbors(rel_type
        Design 2:
            more general like conv, read others implementation
            input_labels: batch_size * (num_context,)
            pre-compute all edge embedding, too much memory(200M edge)
            [],[]
            A[word_id]
        """
        [batch_size, window_size] = input_labels.size()
        out_labels.squeeze_()
        use_cuda = self.out_embed.weight.is_cuda
        #print(input_labels[:, -1], self.num_docs)
        relational_mask = t.ge(input_labels[:, -1], self.num_docs)
        context_idx = t.nonzero(1 - relational_mask).squeeze().view(-1)
        sub_idx = t.nonzero(relational_mask).squeeze().view(-1)

        doc_ids = t.masked_select(input_labels[:, -1], 1 - relational_mask)
        
        # bug
        # num_doc x 
        context_ids = input_labels[:, :-1].index_select(0, context_idx)

        if self.weights is not None:
            noise_sample_count = batch_size * num_sampled
            draw = self.sample(noise_sample_count)
            noise = draw.view(batch_size, num_sampled)
        else:
            noise = Variable(t.Tensor(batch_size, num_sampled).
                             uniform_(0, self.num_words - 1).long())
        #print(t.masked_select(out_labels[:,0], 1 - relational_mask).shape)
        target_labels = t.masked_select(out_labels, 1 - relational_mask).unsqueeze(1)
        target_noise_ids = t.cat((target_labels, noise[:doc_ids.size()[0], :]), dim=1)
        
        if use_cuda:
            doc_ids = doc_ids.cuda()
            context_ids = context_ids.cuda()
            target_noise_ids = target_noise_ids.cuda()
        # relational bias loss and DM loss

        # concatenate version
        # x = t.cat(self.doc_embed[doc_ids, :], torch.sum(self.word_embed[context_ids, :], dim=1))
        # average version
        x = t.add(
            self.doc_embed(doc_ids), t.sum(self.word_embed(context_ids), dim=1))

        output = t.bmm(
            x.unsqueeze(1),
            self.out_embed(target_noise_ids).permute(0, 2, 1)).squeeze()

        if self.relational_bias:
            if relational_mask.sum() == 0:
                return self.nce_loss(output), 0.0
            sub_ids = input_labels[:, 0].index_select(0, sub_idx).cuda()
            r_types = t.masked_select(input_labels[:, -1], relational_mask).cuda() - (self.num_docs+1)
            obj_ids = out_labels.masked_select(relational_mask).unsqueeze(1)
            relation_noise_ids = t.cat((obj_ids, noise[doc_ids.size()[0]:, :]), dim=1).cuda()
            x_proj = self.W_r(self.word_embed(sub_ids)).view(-1, self.kb_emb_size)
            y_proj = self.W_r(self.word_embed(relation_noise_ids)).view(-1, num_sampled + 1, self.kb_emb_size)
            r_x = self.r_linear(r_types, x_proj)
            r_output = t.bmm(
                r_x.unsqueeze(1),
                y_proj.permute(0, 2, 1)).squeeze(dim=1)
            #debug
            #assert r_output.size()[1] == 6
            #return self.nce_loss(r_output)
            
            return self.nce_loss(output), self.nce_loss(r_output).float()
        
        return self.nce_loss(output), 0.0

    def input_embeddings(self):
        return self.word_embed.weight.data.cpu().numpy()

    def doc_embeddings(self):
        return self.doc_embed.weight.data.cpu().numpy()