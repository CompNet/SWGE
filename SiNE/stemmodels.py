import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
#from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import SiNEmaster.util as util  # Altered code.
import numpy as np


# From https://github.com/CompNet/SignedCentrality


def regularize(parameter, p=2):
    zeros = torch.zeros_like(parameter)
    diff = torch.abs(parameter - zeros)
    norm = torch.norm(diff, p)
    return norm


class GraphEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_embedding(self, x):
        raise NotImplementedError

    def get_all_weights(self):
        raise NotImplementedError

    def get_edge_features(self, x, y, operation='hadamard'):
        func = util.FEATURE_FUNCS[operation]
        x_emb = self.get_embedding(x)
        y_emb = self.get_embedding(y)
        return func(x_emb, y_emb)

    def regularize(self, lam=0.0055, p=2):
        regularizer_term = Variable(torch.zeros(1))
        for parameter in self.parameters():
            regularizer_term += regularize(parameter, p)
        regularizer_term *= lam
        return regularizer_term




class EnergyToProbsLayer(nn.Module):
    def __init__(self):
        super(EnergyToProbsLayer, self).__init__()
        self.transform = nn.Sigmoid()
    def forward(self, x):
        ones = torch.ones_like(x)
        positive_prob = self.transform(x)
        negative_prob = ones - positive_prob
        output = torch.cat((negative_prob, positive_prob), dim=1)
        #output = torch.log(output)
        return output




#-----------------------------------------------------------------------------------------------------------------
#
#  SiNE model of Wang et al.
#
#-----------------------------------------------------------------------------------------------------------------



class SiNESubModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.tanh = nn.Tanh()

        initrange = np.sqrt(6.0/(input_dim + output_dim))
        self.layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, input1):
        x = self.layer(input1)
        x += self.bias
        x = self.tanh(x)
        return x


class SiNECompModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, output_dim, bias=False)
        self.layer2 = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.tanh = nn.Tanh()

        initrange = np.sqrt(6.0 / (input_dim + output_dim))
        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer2.weight.data.uniform_(-initrange, initrange)

    def forward(self, input1, input2):
        x = self.layer1(input1) + self.layer2(input2)
        x += self.bias
        x = self.tanh(x)
        return x






class SiNE(GraphEmbeddingModel):
    def __init__(self, num_nodes, dims_arr):
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes + 1, dims_arr[0])
        self.embeddings.weight.data.uniform_(-0.0, 0)
        self.layers = []
        self.comp_layer = SiNECompModule(dims_arr[0], dims_arr[1])
        length = len(dims_arr)
        for i in range(1, length - 1):
            layer = SiNESubModule(dims_arr[i], dims_arr[i + 1])
            self.add_module('l{0}'.format(i), layer)
            self.layers.append(layer)
        layer = SiNESubModule(dims_arr[-1], 1)
        self.add_module('l{0}'.format(len(dims_arr)), layer)
        self.layers.append(layer)

    def get_all_weights(self):
        res = self.embeddings.weight.data.numpy()
        return res

    def forward(self, xi, xj, xk, delta):
        i_emb = self.embeddings(xi)
        j_emb = self.embeddings(xj)
        k_emb = self.embeddings(xk)

        zn1 = self.comp_layer(i_emb, j_emb)
        zn2 = self.comp_layer(i_emb, k_emb)

        for layer in self.layers:
            zn1 = layer(zn1)
            zn2 = layer(zn2)

        f_pos = zn1
        f_neg = zn2

        zeros = Variable(torch.zeros(1))

        loss = torch.max(zeros, f_neg + delta - f_pos)
        loss = torch.sum(loss)
        loss = torch.tensor([loss])  # Added code.

        return loss

    def get_embedding(self, x):
        x = Variable(torch.LongTensor([int(x)]))
        emb = self.embeddings(x)
        emb = emb.data.numpy()[0]
        return emb

    def get_x(self):                                        # Added code.
        return self.layers[-1].layer.weight.data            # Added code.



def fit_sine_model(num_nodes, dims_arr, triples, triples0, delta, delta0, batch_size, batch_size0, epochs,
                   lr=0.01, lam=0.0055, lr_decay=0.0, p=2, print_loss=True, p0=True):
    sine = SiNE(num_nodes, dims_arr)
    optimizer = optim.Adagrad(sine.parameters(), lr=lr, lr_decay=lr_decay)
    for epoch in range(epochs):
        optimizer.zero_grad()
        C = batch_size
        xi, xj, xk = util.get_triples_training_batch(triples, batch_size)
        loss = sine(xi, xj, xk, delta)
        if p0:
            xi, xj, xk = util.get_triples_training_batch(triples0, batch_size0)
            loss += sine(xi, xj, xk, delta0)
            C += batch_size0
        loss /= C
        loss += sine.regularize(lam, p)
        loss.backward()
        optimizer.step()
        if print_loss:
            print('Loss at epoch ', epoch + 1, ' is ', loss.data[0])
    return sine

