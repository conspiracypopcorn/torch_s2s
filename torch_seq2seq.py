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
# enc_emb_size = 10
# dec_emb_size = 10
# hidden_units = 10
# slu_hidden_units = 10
# max_grad_norm = 0.25
# torch.manual_seed(1)
# use_cuda = torch.cuda.is_available()
#
# # dec input -> GO + target
# # dec target -> target + EOS
#
# # Turn a Unicode string to plain ASCII, thanks to
# # http://stackoverflow.com/a/518232/2809427
# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#     )
#
# # Lowercase, trim, and remove non-letter characters
#
#
# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s
#
#
# training_data = [
#     ("Oggi sono stato al mare", "Today I have been at the seaside", "Tb Vb Vi Lb Li"),
#     ("Domani vado a vedere un film in citt", "Tomorrow I'm going to the cinema to see a movie", "Tb Vb Vi Vi Ob Oi Lb Li"),
#     ("Oggi sono andato al lago", "Today I went to the lake", "Tb Vb Vi Lb Li")
# ]
#
# it_to_ix = {}
# eng_to_ix = {}
# pos_to_ix = {}
#
# for it_seq, eng_seq, pos_seq in training_data:
#     it = normalizeString(it_seq).split()
#     eng = normalizeString(eng_seq).split()
#     pos = normalizeString(pos_seq).split()
#     for word in it:
#         if word not in it_to_ix:
#             it_to_ix[word] = len(it_to_ix)
#     for word in eng:
#         if word not in eng_to_ix:
#             eng_to_ix[word] = len(eng_to_ix)
#     for word in pos:
#         if word not in pos_to_ix:
#             pos_to_ix[word] = len(pos_to_ix)
#
# ix_to_it = dict(zip(it_to_ix.values(), it_to_ix.keys()))
# ix_to_eng = dict(zip(eng_to_ix.values(), eng_to_ix.keys()))
# ix_to_pos = dict(zip(pos_to_ix.values(), pos_to_ix.keys()))
#
# enc_voc_size = len(it_to_ix)
# dec_voc_size = len(eng_to_ix)
# pos_voc_size = len(pos_to_ix)
#
# def prepare_sequence(seq, word_to_ix):
#     idxs = [word_to_ix[w] for w in normalizeString(seq).split()]
#     tensor = torch.LongTensor(idxs)
#     result = autograd.Variable(tensor)
#     if use_cuda:
#         return result.cuda()
#     else:
#         return result
#
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
#
#
# decoders = OrderedDict()
#
# decoders["slu"] = pos_voc_size
# decoders["prev"] = dec_voc_size
# decoders["next"] = dec_voc_size
#
#
# model = h.Seq2SeqModel(enc_voc_size, enc_emb_size, hidden_units, decoders,
#                        dec_emb_size, slu_hidden_units)
# if use_cuda:
#     model = model.cuda()
#
# print(model)
#
# optimizer = optim.Adagrad(model.parameters(), lr=0.1)
#
# enc_input = prepare_sequence(training_data[0][0], it_to_ix)
# dec_input = prepare_sequence(training_data[0][1], eng_to_ix)
#
#
# word_scores = model.forward(enc_input, dec_input)
#
# print("input: " + training_data[0][0])
#print_predictions(word_scores)

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
        "prev": prev1,
        "next": next1
    }
    scores = model.forward(inp1, dec_inputs1)
    # print("Input: " + h.idxs_to_string(inp1[0], g.enc_idx2word))
    # print("Slu targ: " + h.idxs_to_string(slu1[0], g.slu_idx2word))
    # print("Prev targ: " + h.idxs_to_string(prev1[0], g.dec_idx2word))
    # print("Next targ: " + h.idxs_to_string(next1[0], g.dec_idx2word))


    print_predictions(scores, g.dec_idx2word, g.slu_idx2word)

    optimizer = optim.Adagrad(model.parameters(), lr=g.learning_rate)
    start = time.time()
    for epoch in range(100):
        epoch_loss = []
        train_batch_generator.reset()
        #  while train_batch_generator.elements_available():
        for _ in range(100):
            model.zero_grad()

            inp, prev, next, slu = train_batch_generator.get_input_prev_next_slu(g.use_cuda)
            # for the moment dec_inputs == dec targets (just for testing)
            dec_inputs = {
                "slu": slu,
                "prev": prev,
                "next": next
            }
            dec_targets = {
                "slu": slu.view(-1),
                "prev": prev.view(-1),
                "next": next.view(-1)
            }
            scores = model.forward(inp, dec_inputs)
            loss = model.loss(scores, dec_targets)
            epoch_loss.append(loss.cpu().data.numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optimizer.step()
        print("Avg loss: {0:.2f}".format(np.mean(epoch_loss)))

    print("Execution Time: {}".format(time.time() - start))
    scores = model.forward(inp1, dec_inputs1)
    print_predictions(scores, g.dec_idx2word, g.slu_idx2word)

# start = time.time()
# for epoch in range(100):
#     batch_loss = []
#     for it_s, eng_s, pos_s in training_data:
#         # 1) clear gradients
#         model.zero_grad()
#         # reset hidden state
#         #model.init_hidden()
#
#         # 2) get inputs ready
#         enc_input = prepare_sequence(it_s, it_to_ix)
#         dec_input = prepare_sequence(eng_s, eng_to_ix)
#         targets = prepare_sequence(eng_s, eng_to_ix)
#         pos_target = prepare_sequence(pos_s, pos_to_ix)
#
#         dec_targ = {"slu": pos_target,
#                     "prev": targets,
#                     "next": targets}
#
#         # 3) run forward pass
#         scores = model.forward(enc_input, dec_input)
#         # 4) loss, gradients, update
#         loss = model.loss(scores, dec_targ)
#         batch_loss.append(loss.data.numpy())
#         loss.backward()
#         torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
#         optimizer.step()
#     if epoch % 10 == 0:
#         print("Avg loss: {0:.2f}".format(np.mean(batch_loss)))
#
# print("Execution Time: {}".format(time.time() - start))
# enc_input = prepare_sequence(training_data[0][0], it_to_ix)
# dec_input = prepare_sequence(training_data[0][1], eng_to_ix)
#
# #enc_out, enc_final_state = enc.forward(enc_input)
#
# word_scores = model.forward(enc_input, dec_input)
#
# print("input: " + training_data[0][0])
# print_predictions(word_scores)