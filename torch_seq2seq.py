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
import os
import shutil
# # dec input -> GO + target
# # dec target -> target + EOS
# da fare:
# 1: feed previous X
# 2: dev phase X
# 3: early stopping/lr decay X
# 4: test phase
# 5: save/load model
def print_predictions(word_scores, dec_dict=None, slu_dict=None):
    for name in word_scores.keys():
        w_cpu = word_scores[name].cpu()
        w_s = np.squeeze(w_cpu.data.numpy())
        pred = np.argmax(w_s, axis=1)
        if name == "slu":
            trad = [slu_dict[i] for i in list(pred)]
        else:
            trad = [dec_dict[i] for i in list(pred[:-1])] # remove eos tag
        print(name + ": " + " ".join(trad))

def log_predictions(word_scores, test_path, dec_dict=None, slu_dict=None):
    for name in word_scores.keys():
        w_cpu = word_scores[name].cpu()
        w_s = w_cpu.data.numpy()
        pred = np.argmax(w_s, axis=1)
        if name == "slu":
            trad = [slu_dict[i] for i in list(pred)]
        else:
            trad = [dec_dict[i] for i in list(pred[:-1])]
        with open(test_path + name + ".txt", "a") as f:
            f.write(" ".join(trad) +"\n")

def train(g):
    lr = g.learning_rate
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

    dev_batch_generator = h.batch_gen(g.batch_size, g.dev_user_data, g.dev_prev_system_data,
                                      g.dev_next_system_data, g.dev_slu_data)

    inp1, prev1, next1, slu1 = train_batch_generator.get_input_prev_next_slu(g.use_cuda)

    dec_inputs1 = {
        "slu": slu1,
        "prev": torch.cat([h.go_tag(g.use_cuda),prev1], 0),
        "next": torch.cat([h.go_tag(g.use_cuda), next1], 0)
    }
    scores = model.forward(inp1, dec_inputs1)
    print_predictions(scores, g.dec_idx2word, g.slu_idx2word)

    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    start = time.time()

    dev_loss_track = []
    for epoch in range(50):
        epoch_loss = []
        train_batch_generator.reset()

        #  while train_batch_generator.elements_available():
        for _ in range(1000):
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

        print("Avg epoch loss: {0:.2f}".format(np.mean(epoch_loss)))

        # dev phase
        dev_batch_generator.reset()
        dev_loss = []
        while dev_batch_generator.elements_available():
            inp, prev, next, slu = dev_batch_generator.get_input_prev_next_slu(g.use_cuda)
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
            dev_loss.append(loss.cpu().data.numpy())
        avg_dev_loss = np.mean(dev_loss)
        print("Avg Dev Loss: {0:.2f}".format(avg_dev_loss))
        dev_loss_track.append(avg_dev_loss)
        # lr decay ~ early stopping
        if avg_dev_loss == min(dev_loss_track):
            # save model
            print("Save Model")
            if not os.path.exists(g.s2s_model_path):
                os.makedirs(g.s2s_model_path)
            torch.save(model.state_dict(), g.s2s_model_path + "model.s2s")
        else:
            lr *= g.learning_rate_decay_factor
            print("Lr decay, LR = {:.2f}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if (len(dev_loss_track) - np.argmin(dev_loss_track)) > 3:
                print("Training ended for early stopping (3 consecutive decay).")
                break
            if lr < g.learning_rate/256:
                print("Training ended for early stoppin (min lr reached).")
                break

    print("Execution Time: {}".format(time.time() - start))
    model.feed_previous = True
    dec_inputs1 = {
        "slu": slu1,
        "prev": h.go_tag(g.use_cuda),
        "next": h.go_tag(g.use_cuda)
    }
    scores = model.forward(inp1, dec_inputs1)
    print_predictions(scores, g.dec_idx2word, g.slu_idx2word)

def test(g):
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
        slu_hidden_units=g.slu_hidden_units,
        feed_previous=True
    )

    model.load_state_dict(torch.load(g.s2s_model_path + "model.s2s"))

    test_batch_generator = h.batch_gen(g.batch_size, g.test_user_data, g.test_prev_system_data,
                                      g.test_next_system_data, g.test_slu_data)
    start = time.time()
    if os.path.exists(g.s2s_test_path):
        shutil.rmtree(g.s2s_test_path, ignore_errors=True)
    os.makedirs(g.s2s_test_path)


    while test_batch_generator.elements_available():
        inp, prev, next, slu = test_batch_generator.get_input_prev_next_slu(g.use_cuda)
        # for the moment dec_inputs == dec targets (just for testing)
        dec_inputs = {
            "slu": slu,  # this is not used :D
            "prev": h.go_tag(g.use_cuda),
            "next": h.go_tag(g.use_cuda)
        }

        scores = model.forward(inp, dec_inputs)

        log_predictions(scores, g.s2s_test_path, g.dec_idx2word, g.slu_idx2word)

    if g.prev_t:
        with open(g.data_dir + "test_prev_system_turns.txt", "r") as fi:
            with open(g.s2s_test_path + "prev_targ.txt", "w") as fo:
                lines = fi.readlines()
                for line in lines:
                    line = line.strip()
                    if line != "":
                        fo.write(line + "\n")
    if g.next_t:
        with open(g.data_dir + "test_next_system_turns.txt", "r") as fi:
            with open(g.s2s_test_path + "next_targ.txt", "w") as fo:
                lines = fi.readlines()
                for line in lines:
                    line = line.strip()
                    if line != "":
                        fo.write(line + "\n")
    if g.slu:
        with open(g.data_dir + "test_slu_turns.txt", "r") as fi:
            with open(g.s2s_test_path + "slu_targ.txt", "w") as fo:
                lines = fi.readlines()
                for line in lines:
                    line = line.strip()
                    if line != "":
                        fo.write(line + "\n")

    print("time {}".format(time.time() - start))

