# -*- coding: utf-8 -*-
import torch_helpers as h
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import re
import unicodedata
import torch
import numpy as np
import time
from collections import OrderedDict
# # dec input -> GO + target
# # dec target -> target + EOS
# da fare:
# 1: feed previous
# - quando uso il flag feed_previous, do in input ai decoder solo il  [go]
# - itera finchè non arriva [eos]
# 2: dev phase
# 3: early stopping/lr decay
# 4: test phase
def print_predictions(word_scores, dec_dict=None, slu_dict=None):
    for name in word_scores.keys():
        w_cpu = word_scores[name].cpu()
        w_s = np.squeeze(w_cpu.data.numpy())
        pred = np.argmax(w_s, axis=1)
        if name == "slu":
            trad = [slu_dict[i] for i in list(pred)]
        else:
            trad = [dec_dict[i] for i in list(pred)]
        print(name + ": " + " ".join(trad))


def train(g):

    decoders = OrderedDict()
    if g.slu:
        decoders["slu"] = g.slu_vocab_size
    if g.prev_t:
        decoders["prev"] = g.dec_vocab_size
    if g.next_t:
        decoders["next"] = g.dec_vocab_size

    model = h.Seq2SeqModel(
        enc_voc_size=g.enc_vocab_size,
        enc_emb_size=g.enc_embedding_size,
        hidden_units=g.hidden_units,
        decoders=decoders,
        dec_emb_size=g.dec_embedding_size,
        slu_hidden_units=g.slu_hidden_units
    )

    if g.use_cuda:
        model = model.cuda()

    print(model)

    train_batch_generator = h.batch_gen(g.batch_size, g.train_user_data, g.train_prev_system_data,
                                        g.train_next_system_data, g.train_slu_data)

    inp1, prev1, next1, slu1 = train_batch_generator.get_input_prev_next_slu(g.use_cuda)

    dec_inputs1 = {
        "slu": slu1,
        "prev": torch.cat([h.go_tag(g.use_cuda),prev1], 0),
        "next": torch.cat([h.go_tag(g.use_cuda), next1], 0)
    }
    scores = model.forward(inp1, dec_inputs1)
    print_predictions(scores, g.dec_idx2word, g.slu_idx2word)

    optimizer = optim.Adagrad(model.parameters(), lr=g.learning_rate)
    start = time.time()
    for epoch in range(50):
        epoch_loss = []
        train_batch_generator.reset()
        #  while train_batch_generator.elements_available():
        for _ in range(100):
            model.zero_grad()

            inp, prev, next, slu = train_batch_generator.get_input_prev_next_slu(g.use_cuda)
            # for the moment dec_inputs == dec targets (just for testing)
            dec_inputs = {
                "slu": slu,  # this is not used :D
                "prev": torch.cat([h.go_tag(g.use_cuda), prev], 0),
                "next": torch.cat([h.go_tag(g.use_cuda), next], 0)
            }
            dec_targets = {
                "slu": slu.view(-1),
                "prev": torch.cat([prev, h.eos_tag(g.use_cuda)], 0).view(-1),
                "next": torch.cat([next, h.eos_tag(g.use_cuda)], 0).view(-1)
            }
            scores = model.forward(inp, dec_inputs)
            loss = model.loss(scores, dec_targets)
            epoch_loss.append(loss.cpu().data.numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optimizer.step()
        print("Avg loss: {0:.2f}".format(np.mean(epoch_loss)))

    print("Execution Time: {}".format(time.time() - start))
    model.feed_previous = True
    dec_inputs1 = {
        "slu": slu1,
        "prev": h.go_tag(g.use_cuda),
        "next": h.go_tag(g.use_cuda)
    }
    scores = model.forward(inp1, dec_inputs1)
    print_predictions(scores, g.dec_idx2word, g.slu_idx2word)