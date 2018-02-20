import numpy as np
import random
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

PAD = 0
GO = 1
EOS = 2
use_cuda = False

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

'''
Read a vocabulary file from the location file_path and return 2 dictionaries respectively from word to index and from index to word
'''
def read_dictionary(file_path):
    word2idx = {}
    word_dict = open(file_path,"r")
    for line in word_dict:
        key,val = line.split()
        word2idx[key] = int(val)
    idx2word = dict((k,v) for v,k in word2idx.items())
    return (word2idx, idx2word)

'''
Read the data file from file_path location and substitute all the words with the respective indexes as provided by the dictionary words2idx. Return all the sequences in a list of lists
'''
def read_data(file_path, word2idx):
    sequences_list = []
    sequence = []
    with open(file_path, "r") as turns:
        for line in turns:
            if line.isspace():
                sequences_list.append(sequence)
                sequence = []
            else:
                tokens = line.split()
                tokens_idxs = []
                for token in tokens:
                    if token in word2idx.keys():
                        tokens_idxs.append(word2idx[token])
                    else:
                        tokens_idxs.append(word2idx["<unk>"])
                sequence.append(tokens_idxs)
        sequences_list.append(sequence)
    return sequences_list


'''take a list of indexes and convert it into a string of tokens, removing eos and pad after the first'''
def idxs_to_string(idxs, idx2word):
    tokens = []
    for idx in idxs:
        if idx == EOS:
            break
        else:
            tokens.append(idx2word[idx])
    return " ".join(tokens)

def idxs_to_list(idxs, idx2word):
    tokens = []
    for idx in idxs:
        if idx == EOS:
            break
        else:
            tokens.append(idx2word[idx])
    return tokens

def flatten(a):
    for dim in np.shape(a):
        np.resize(a,-1)
    return a

def go_tag(use_cuda, batch_size = 1):
    got = torch.LongTensor(np.full((1,batch_size), GO))
    if use_cuda:
        return Variable(got).cuda()
    else:
        return Variable(got)

def eos_tag(use_cuda, batch_size = 1):
    got = torch.LongTensor(np.full((1,batch_size), EOS))
    if use_cuda:
        return Variable(got).cuda()
    else:
        return Variable(got)

class batch_gen():

    def __init__(self, batch_size, input_sequences_list, target_prev_list,target_next_list, slu_data):
        self.batch_size = batch_size
        self.tuple_list = self.make_tuple_list(input_sequences_list,
                                               target_prev_list,
                                               target_next_list,
                                               slu_data)
        self.turns_number = len(self.tuple_list)
        self.curr = 0

    def make_tuple_list(self, input, targ_prev, targ_next, slu):
        tuple_list = []
        # check the size of the data is correct
        for in_seq, p_targ_seq, n_targ_seq, slu_seq in zip(input, targ_prev, targ_next, slu):
            assert len(in_seq) == len(p_targ_seq) and len(n_targ_seq) == len(slu_seq) and len(in_seq) == len(slu_seq)
            for i_t, p_t, n_t, s_t in zip(in_seq, p_targ_seq, n_targ_seq, slu_seq):
                tuple_list.append((i_t, p_t, n_t, s_t))
        return tuple_list

    def elements_available(self):
        if self.curr + self.batch_size <= self.turns_number:
            return True
        else:
            return False

    def shuffle(self):
        self.curr = 0
        random.shuffle(self.tuple_list)

    def reset(self):
        self.curr = 0

    def get_input_prev_next_slu(self, use_cuda):
        input_batch = []
        prev_target_batch = []
        next_target_batch = []
        slu_target_batch = []
        if self.elements_available():
            for tuple in self.tuple_list[self.curr:(self.curr + self.batch_size)]:
                input_batch.append(tuple[0])
                prev_target_batch.append(tuple[1])
                next_target_batch.append(tuple[2])
                slu_target_batch.append(tuple[3])
        self.curr += self.batch_size

        if use_cuda:
            return (Variable(torch.LongTensor(input_batch).transpose(0,1)).cuda(),
                    Variable(torch.LongTensor(prev_target_batch).transpose(0,1).cuda()),
                    Variable(torch.LongTensor(next_target_batch).transpose(0, 1).cuda()),
                    Variable(torch.LongTensor(slu_target_batch).transpose(0, 1).cuda()))
        else:
            return (Variable(torch.LongTensor(input_batch).transpose(0,1)),
                    Variable(torch.LongTensor(prev_target_batch).transpose(0,1)),
                    Variable(torch.LongTensor(next_target_batch).transpose(0, 1)),
                    Variable(torch.LongTensor(slu_target_batch).transpose(0, 1)))

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_units):
        super(EncoderRNN, self).__init__()
        self.hidden_units = hidden_units
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_units)


    def forward(self, input):
        input_embedded = self.embedding(input).view(len(input), 1, -1)
        lstm_out, final_state = self.lstm(input_embedded) # if no init-state is defined it's zero by default
        return lstm_out, final_state


class DecoderRnn(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_units):
        super(DecoderRnn, self).__init__()
        self.hidden_units = hidden_units
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_units)

    def forward(self, input, enc_final_state):
        embedded = self.embedding(input).view(len(input), 1, -1)
        lstm_out, final_state = self.lstm(embedded, enc_final_state)
        return lstm_out, final_state


class SluDecoderRnn(nn.Module):
    def __init__(self, voc_size, hidden_units, enc_hidden_units):
        super(SluDecoderRnn, self).__init__()
        self.hidden_units = hidden_units
        self.voc_size = voc_size
        self.lstm = nn.LSTM(input_size=enc_hidden_units, hidden_size=self.hidden_units, bidirectional=True)
        self.linear = nn.Linear(self.hidden_units*2, voc_size)

    def forward(self, inputs):
        # inputs is the output of the encoder : shape(time, batch, enc_hidden_units)
        output, _ = self.lstm(inputs) # (time, batch, hidden)
        flat_out = output.view(-1, self.hidden_units*2) # time*batch, hidden
        logits = self.linear(flat_out) # time*batch, voc
        tag_scores = F.log_softmax(logits, dim=1) # time*batch , voc
        return tag_scores

class Seq2SeqModel(nn.Module):
    def __init__(self, enc_voc_size, enc_emb_size, hidden_units, decoders, dec_emb_size,
                 slu_hidden_units, feed_previous=False):
        # dec dict "name" : voc_size
        super(Seq2SeqModel, self).__init__()
        self.enc = EncoderRNN(enc_voc_size, enc_emb_size, hidden_units)
        self.feed_previous = feed_previous
        self.decoders = nn.ModuleList()
        self.dec_names = list(decoders.keys()) # need for order
        self.hidden_units = hidden_units
        slu_only = True
        for name in self.dec_names:
            if name == "slu":
                self.decoders.append(SluDecoderRnn(decoders[name], slu_hidden_units, hidden_units))
            else:
                self.decoders.append(DecoderRnn(decoders[name], dec_emb_size, hidden_units))
                slu_only = False
                dec_voc_size = decoders[name]
        if not slu_only:
            self.linear = nn.Linear(hidden_units, dec_voc_size)
        self.loss_function = nn.NLLLoss()

    def forward(self, enc_input, dec_inputs=None):
        scores = OrderedDict()
        enc_out, enc_final_state = self.enc.forward(enc_input)

        for dec, name in zip(self.decoders, self.dec_names):
            if name == "slu":
                scores[name] = dec.forward(enc_out)
            else:
                if self.feed_previous:
                    # dec-inputs = [go]
                    logits_list = []
                    assert len(dec_inputs[name]) == 1

                    def get_pred(logits): # t*b, voc

                        _, pred = torch.max(logits, dim=1) # t*b
                        return pred.view(-1, 1)

                    time = 0
                    out, hidden_state = dec.forward(dec_inputs[name], enc_final_state) # t, b , hid
                    out_flat = out.view(-1, self.hidden_units)
                    logits = self.linear(out_flat)
                    logits_list.append(logits)
                    pred = get_pred(logits)

                    while (pred != EOS).any() and time < 100:
                        out, hidden_state = dec.forward(pred, hidden_state)
                        out_flat = out.view(-1, self.hidden_units)
                        logits = self.linear(out_flat)
                        pred = get_pred(logits)
                        logits_list.append(logits)
                        time += 1

                    logits = torch.cat(logits_list, 0)
                    scores[name] = F.log_softmax(logits, dim=1)

                else:
                    out, _ = dec.forward(dec_inputs[name], enc_final_state) # [time * bs, hidden]
                    logits = self.linear(out.view(-1,self.hidden_units))
                    scores[name] = F.log_softmax(logits, dim=1)
        return scores

    def loss(self, scores, targets):
        losses = []
        for name in targets.keys():
                l = self.loss_function(scores[name], targets[name])
                losses.append(l)
        return torch.mean(torch.cat(losses, 0))
