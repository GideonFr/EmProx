# Author: Yang Liu @ Abacus.ai
# This is an implementation of the semi-supervised predictor for NAS from the paper:
# Luo et al., 2020. "Semi-Supervised Neural Architecture Search" https://arxiv.org/abs/2002.10389
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import itertools
import os
import random
import sys
import math
import numpy as np
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from scipy.spatial.distance import cdist

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from naslib.utils.utils import AverageMeterGroup, AverageMeter
from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor
from naslib.predictors.trees.ngb import loguniform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# TO RUN
# python runner.py --config-file naslib/benchmarks/predictors/predictor_config.yaml

# default parameters from the paper
n = 1100
# m = 10000
nodes = 8
new_arch = 300
k = 100
encoder_layers = 1
hidden_size = 128 #64 
mlp_layers = 2
mlp_hidden_size = 16
decoder_layers = 1
source_length = 35  # 27
encoder_length = 35  # 27
decoder_length = 35  # 27
dropout = 0.1
l2_reg = 1e-4
vocab_size = 9  # 7 original
max_step_size = 100
trade_off = 0.8
up_sample_ratio = 10
batch_size = 100
lr = 0.001
optimizer = "adam"
grad_bound = 5.0
# iterations = 3

use_cuda = True
# pretrain_epochs = 1000
# epochs = 1000


def move_to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def convert_arch_to_seq(matrix, ops, max_n=8):
    seq = []
    n = len(matrix)
    max_n = max_n
    assert n == len(ops)
    for col in range(1, max_n):
        if col >= n:
            seq += [0 for i in range(col)]
            seq.append(0)
        else:
            for row in range(col):
                seq.append(matrix[row][col] + 1)
            seq.append(ops[col] + 2)

    assert len(seq) == (max_n + 2) * (max_n - 1) / 2
    return seq


def convert_seq_to_arch(seq):
    n = int(math.floor(math.sqrt((len(seq) + 1) * 2)))
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    ops = [0]
    for i in range(n - 1):
        offset = (i + 3) * i // 2
        for j in range(i + 1):
            matrix[j][i + 1] = seq[offset + j] - 1
        ops.append(seq[offset + i + 1] - 2)
    return matrix, ops


class ControllerDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=None, train=True, sos_id=0, eos_id=0):
        super(ControllerDataset, self).__init__()
        if targets is not None:
            assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id

    def __getitem__(self, index):
        encoder_input = self.inputs[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = [self.targets[index]]
        if self.train:
            decoder_input = [self.sos_id] + encoder_input[:-1]
            sample = {
                "encoder_input": torch.LongTensor(encoder_input),
                "encoder_target": torch.FloatTensor(encoder_target),
                "decoder_input": torch.LongTensor(decoder_input),
                "decoder_target": torch.LongTensor(encoder_input),
            }
        else:
            sample = {
                "encoder_input": torch.LongTensor(encoder_input),
                "decoder_target": torch.LongTensor(encoder_input),
            }
            if encoder_target is not None:
                sample["encoder_target"] = torch.FloatTensor(encoder_target)
        return sample

    def __len__(self):
        return len(self.inputs)


class Encoder(nn.Module):
    def __init__(
        self,
        layers,
        mlp_layers,
        hidden_size,
        mlp_hidden_size,
        vocab_size,
        dropout,
        source_length,
        length,
    ):
        super(Encoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # self.mlp_layers = mlp_layers
        # self.mlp_hidden_size = mlp_hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = dropout
        self.rnn = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.out_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)


    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, self.dropout, training=self.training)
        residual = x
        x, hidden = self.rnn(x)
        x = self.out_proj(x)
        x = residual + x
        x = F.normalize(x, 2, dim=-1)
        encoder_outputs = x
        encoder_hidden = hidden

        x = torch.mean(x, dim=1)
        x = F.normalize(x, 2, dim=-1)
        arch_emb = x

        return encoder_outputs, encoder_hidden, arch_emb #, predict_value


SOS_ID = 0
EOS_ID = 0


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)

    def forward(self, input, source_hids, mask=None):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if mask is not None:
            attn.data.masked_fill_(mask, -float("inf"))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(
            batch_size, -1, source_len
        )

        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)

        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(
            self.output_proj(combined.view(-1, self.input_dim + self.source_dim))
        ).view(batch_size, -1, self.output_dim)

        return output, attn


class Decoder(nn.Module):
    def __init__(
        self,
        layers,
        hidden_size,
        vocab_size,
        dropout,
        length,
    ):
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length
        self.vocab_size = vocab_size
        self.rnn = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.layers,
            batch_first=True,
            dropout=dropout,
        )
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.init_input = None
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = dropout
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.n = int(math.floor(math.sqrt((self.length + 1) * 2)))
        self.offsets = []
        for i in range(self.n):
            self.offsets.append((i + 3) * i // 2 - 1)

    def forward(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_hidden = self._init_state(encoder_hidden)
        if x is not None:
            bsz = x.size(0)
            tgt_len = x.size(1)
            x = self.embedding(x)
            x = F.dropout(x, self.dropout, training=self.training)
            residual = x
            x, hidden = self.rnn(x, decoder_hidden)
            x = (residual + x) * math.sqrt(0.5)
            residual = x
            x, _ = self.attention(x, encoder_outputs)
            x = (residual + x) * math.sqrt(0.5)
            predicted_softmax = F.log_softmax(
                self.out(x.view(-1, self.hidden_size)), dim=-1
            )
            predicted_softmax = predicted_softmax.view(bsz, tgt_len, -1)
            return predicted_softmax, None

        # inference
        assert x is None
        bsz = encoder_hidden[0].size(1)
        length = self.length
        decoder_input = encoder_hidden[0].new(bsz, 1).fill_(0).long()
        decoded_ids = encoder_hidden[0].new(bsz, 0).fill_(0).long()

        def decode(step, output):
            if step in self.offsets:  # sample operation, should be in [3, 7]
                symbol = output[:, 3:].topk(1)[1] + 3
            else:  # sample connection, should be in [1, 2]
                symbol = output[:, 1:3].topk(1)[1] + 1
            return symbol

        for i in range(length):
            x = self.embedding(decoder_input[:, i : i + 1])
            x = F.dropout(x, self.dropout, training=self.training)
            residual = x
            x, decoder_hidden = self.rnn(x, decoder_hidden)
            x = (residual + x) * math.sqrt(0.5)
            residual = x
            x, _ = self.attention(x, encoder_outputs)
            x = (residual + x) * math.sqrt(0.5)
            output = self.out(x.squeeze(1))
            symbol = decode(i, output)
            decoded_ids = torch.cat((decoded_ids, symbol), axis=-1)
            decoder_input = torch.cat((decoder_input, symbol), axis=-1)

        return None, decoded_ids

    def _init_state(self, encoder_hidden):
        """Initialize the encoder hidden state."""
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden


class NAO(nn.Module):
    def __init__(
        self,
        encoder_layers,
        decoder_layers,
        mlp_layers,
        hidden_size,
        mlp_hidden_size,
        vocab_size,
        dropout,
        source_length,
        encoder_length,
        decoder_length,
        k_nn
    ):
        super(NAO, self).__init__()
        self.encoder = Encoder(
            encoder_layers,
            mlp_layers,
            hidden_size,
            mlp_hidden_size,
            vocab_size,
            dropout,
            source_length,
            encoder_length,
        ).to(device)
        self.decoder = Decoder(
            decoder_layers,
            hidden_size,
            vocab_size,
            dropout,
            decoder_length,
        ).to(device)
        self.k_nn = k_nn

        self.flatten_parameters()
        self.embedding_list = np.empty((0, hidden_size))
        self.accuracy_list = []

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def estimate_scores(self, K, unknown_archs, known_archs, accuracies):
        # compute distance matrix from all unscored archs to all scored archs 
        dist_mat = cdist(unknown_archs, known_archs, metric='euclidean')
        est_accs = []

        for i, row in enumerate(dist_mat):
            # partition the distances in the distance matrix such that first k are sorted (closest to furthest)
            k_min_dist = np.argpartition(row, (K-1)) # faster than argsort
        
            # determine weight by inverse distance weighting
            # weight = distance / sum(distances), inverse by taking weight = 1/weight
            # alternative calculation inverse weights: 1/dist / sum(1/distances)
            weights = [dist_mat[i][k_min_dist[j]] / (np.sum(np.fromiter(iter=[dist_mat[i][k_min_dist[l]] for l in range(K)], dtype=float))) for j in range(K)]
            temp_weights = [(1/weight) for weight in weights]
            inverse_weights = [temp_weight / (np.sum(temp_weights)) for temp_weight in temp_weights]

            # get accuracies of k nearest architectures
            known_accs = []
            for j in range(K):
                known_accs.append(accuracies[k_min_dist[j]]) 

            # compute score my summing the weights and the accuracies of the corresponding archs
            inverse_score = np.sum(np.multiply(inverse_weights, known_accs))
            est_accs.append(inverse_score)

        return est_accs

    def forward(self, input_variable, target_variable=None):
        if self.training:
            # print(f'input var: {input_variable.shape}')
            encoder_outputs, encoder_hidden, arch_emb = self.encoder( # , predict_value
                input_variable.to(device)
            )

            # print(f'num nan in emb: {np.count_nonzero(~np.isnan(arch_emb.detach().numpy()))}')

            # print(f'embedding: {arch_emb.shape}')
            # print(arch_emb.detach().numpy())
            # self.embedding_list.append(arch_emb.detach().numpy()) # are same archs passed through forward multiple times? Then list invalid
            self.embedding_list = np.concatenate((self.embedding_list, arch_emb.detach().numpy()), axis=0)
            # print(f'embedding list: {shape(self.embedding_list)}')
            # print(f'embedding list element: {shape(self.embedding_list[-1])}')
            # print(self.embedding_list)

            decoder_hidden = (
                arch_emb.unsqueeze(0).to(device),
                arch_emb.unsqueeze(0).to(device),
            )
            decoder_outputs, archs = self.decoder(
                target_variable.to(device), decoder_hidden, encoder_outputs.to(device)
            )
            return decoder_outputs, archs
        else:
            encoder_outputs, encoder_hidden, arch_emb = self.encoder( # , predict_value
                input_variable.to(device)
            )
            # decoder_hidden = (
            #     arch_emb.unsqueeze(0).to(device),
            #     arch_emb.unsqueeze(0).to(device),
            # )
            # decoder_outputs, archs = self.decoder(
            #     target_variable.to(device), decoder_hidden, encoder_outputs.to(device)
            # )           
            unknown_archs = arch_emb.detach().numpy()
            known_archs = np.array(self.embedding_list)
            accuracies = [val.item() for sublist in self.accuracy_list for val in sublist] # flatten list
            predict_value = self.estimate_scores(K=self.k_nn, unknown_archs=unknown_archs, known_archs=known_archs, accuracies=accuracies) 
            return predict_value#, decoder_outputs, archs

    def visualize(self, embeddings, scores, new_mask):
        dim2_emb = TSNE(n_components=2).fit_transform(embeddings)
        scaler = MinMaxScaler()
        dim2_emb_sc = scaler.fit_transform(dim2_emb)
        dim2_emb_sc = [(x, y) for x, y in dim2_emb_sc]
        # make new architectures larger
        # sizes = [100 if i==0 else 20 for i in new_mask]
        plt.scatter(*zip(*dim2_emb_sc), c=scores, cmap='RdYlGn', alpha=0.3) # s=sizes,
        plt.title('Plot embeddings') 
        plt.savefig('plot_embeddings.png')


def controller_train(train_queue, model, optimizer):

    objs = AverageMeter()
    mse = AverageMeter()
    nll = AverageMeter()
    model.train()
    for step, sample in enumerate(train_queue):

        encoder_input = move_to_cuda(sample["encoder_input"])   # architecture
        encoder_target = move_to_cuda(sample["encoder_target"]) # normalized accuracies
        decoder_input = move_to_cuda(sample["decoder_input"])   # embedding?
        decoder_target = move_to_cuda(sample["decoder_target"]) # architecture?

        model.accuracy_list.append(encoder_target.detach().numpy())

        optimizer.zero_grad()
        log_prob, arch = model(encoder_input, decoder_input) # predict_value, 
        # loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze()) # regression loss 
        loss_2 = F.nll_loss( # only use reconstruction loss
            log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1)
        )
        #loss = trade_off * loss_1 + (1 - trade_off) * loss_2
        loss = loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_bound)
        optimizer.step()

        n = encoder_input.size(0)
        objs.update(loss.data, n)
        #mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)

    ################################### TODO PROBLEM FIX #########################################
    # There are always 64 nan values in each row of the embedding list
    # also embedding list instantiated as np.empty((0,64)) since hidden dim = 64
    # coincidence? i think not
    # also:
    # num nan in emb: 6400
    # num nan in emb: 1344
    # num nan in emb: 6400
    # num nan in emb: 1344

    return objs.avg, mse.avg, nll.avg


# def controller_infer(queue, model, step, direction="+"):
#     new_arch_list = []
#     new_predict_values = []
#     model.eval()
#     for i, sample in enumerate(queue):
#         encoder_input = move_to_cuda(sample["encoder_input"])
#         model.zero_grad()
#         new_arch, new_predict_value = model.generate_new_arch(
#             encoder_input, step, direction=direction
#         )
#         new_arch_list.extend(new_arch.data.squeeze().tolist())
#         new_predict_values.extend(new_predict_value.data.squeeze().tolist())
#     return new_arch_list, new_predict_values


def train_controller(model, train_input, train_target, epochs):

    logging.info("Train data: {}".format(len(train_input)))
    controller_train_dataset = ControllerDataset(train_input, train_target, True)
    controller_train_queue = torch.utils.data.DataLoader(
        controller_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    for epoch in range(1, epochs + 1):
        loss, mse, ce = controller_train(controller_train_queue, model, optimizer)
        if epoch % 10 == 0:
            print("epoch {} train loss {} mse {} ce {}".format(epoch, loss, mse, ce))


class EmProxPredictor(Predictor):
    def __init__(
        self,
        k_nn, 
        hidden_lay,
        encoding_type="seminas",
        ss_type=None,
        semi=False,
        hpo_wrapper=False,
        synthetic_factor=1,
    ):
        self.encoding_type = encoding_type
        self.semi = False #semi # OWN CODE FIX TODO
        self.synthetic_factor = synthetic_factor
        if ss_type is not None:
            self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        self.default_hyperparams = {"gcn_hidden": 64, "batch_size": 100, "lr": 1e-3}
        self.hyperparams = None
        self.k_nn = k_nn
        self.hidden_lay = hidden_lay


    def fit(
        self,
        xtrain,
        ytrain,
        train_info=None,
        wd=0,
        iterations=1,
        epochs=50,
        pretrain_epochs=50,
    ):

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        batch_size = self.hyperparams["batch_size"]
        gcn_hidden = self.hyperparams["gcn_hidden"]
        lr = self.hyperparams["lr"]
        up_sample_ratio = 10

        if self.ss_type == "nasbench101":
            self.max_n = 7
            encoder_length = 27
            decoder_length = 27
            vocab_size = 7

        elif self.ss_type == "nasbench201":
            self.max_n = 8
            encoder_length = 35
            decoder_length = 35
            vocab_size = 9

        elif self.ss_type == "darts":
            self.max_n = 35
            encoder_length = 629
            decoder_length = 629
            vocab_size = 13

        elif self.ss_type == "nlp":
            self.max_n = 25
            encoder_length = 324
            decoder_length = 324
            vocab_size = 12
            
        elif self.ss_type == "transbench101":
            self.max_n = 8
            encoder_length = 35
            decoder_length = 35
            vocab_size = 9
            
            
        # get mean and std, normlize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain_normed = (ytrain - self.mean) / self.std
        # encode data in seq
        train_seq_pool = []
        train_target_pool = []
        for i, arch in enumerate(xtrain):
            encoded = encode(
                arch, encoding_type=self.encoding_type, ss_type=self.ss_type
            )
            seq = convert_arch_to_seq(
                encoded["adjacency"], encoded["operations"], max_n=self.max_n
            )
            train_seq_pool.append(seq)
            train_target_pool.append(ytrain_normed[i])

        self.model = NAO(
            encoder_layers,
            decoder_layers,
            mlp_layers,
            self.hidden_lay,
            mlp_hidden_size,
            vocab_size,
            dropout,
            source_length,
            encoder_length,
            decoder_length,
            self.k_nn
        ).to(device)

        for i in range(iterations):
            #print("Iteration {}".format(i + 1))

            train_encoder_input = train_seq_pool
            train_encoder_target = train_target_pool

            # Pre-train
            print("Pre-train EPD")
            train_controller(
                self.model, train_encoder_input, train_encoder_target, pretrain_epochs
            )
            print("Finish pre-training EPD")


        train_pred = np.squeeze(self.query(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))
        return train_error

    def query(self, xtest, info=None, eval_batch_size=100):

        test_seq_pool = []
        for i, arch in enumerate(xtest):
            encoded = encode(
                arch, encoding_type=self.encoding_type, ss_type=self.ss_type
            )
            seq = convert_arch_to_seq(
                encoded["adjacency"], encoded["operations"], max_n=self.max_n
            )
            test_seq_pool.append(seq)

        test_dataset = ControllerDataset(test_seq_pool, None, False)
        test_queue = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        self.model.eval()
        
        # visualize trained embeddings 
        # flat_acc_list = [val.item() for sublist in self.model.accuracy_list for val in sublist] # flatten list
        # self.model.visualize(self.model.embedding_list, flat_acc_list, '')

        pred = []
        with torch.no_grad():
            for _, sample in enumerate(test_queue):
                encoder_input = move_to_cuda(sample["encoder_input"])
                decoder_target = move_to_cuda(sample["decoder_target"]) # the same as enc_input if in eval mode
                prediction = self.model(encoder_input, decoder_target)
                pred.append(prediction)

        pred = np.concatenate(pred)
        return np.squeeze(pred * self.std + self.mean)

    def set_random_hyperparams(self):

        if self.hyperparams is None:
            params = self.default_hyperparams.copy()

        else:
            params = {
                "gcn_hidden": int(loguniform(16, 128)),
                "batch_size": int(loguniform(32, 256)),
                "lr": loguniform(0.00001, 0.1),
            }

        self.hyperparams = params
        return params

    def pre_compute(self, xtrain, xtest, unlabeled):
        """
        This method is used to pass in unlabeled architectures
        for SemiNAS to use, in standalone predictor experiments.
        """
        self.unlabeled = unlabeled

    def set_pre_computations(
        self,
        unlabeled=None,
        xtrain_zc_info=None,
        xtest_zc_info=None,
        unlabeled_zc_info=None,
    ):
        """
        This is the method to pass in unlabeled architectures during
        NAS. The reason we need this method and pre_compute() is to be
        consistent with omni_seminas, where the methods do different things.
        """
        if unlabeled is not None:
            self.unlabeled = unlabeled

    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query, such as a partial learning curve,
        hyperparameters, or how much unlabeled data it needs
        """
        reqs = {
            "requires_partial_lc": False,
            "metric": None,
            "requires_hyperparameters": False,
            "hyperparams": {},
            "unlabeled": self.semi,
            "unlabeled_factor": self.synthetic_factor,
        }
        return reqs
