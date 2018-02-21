import torch_seq2seq
import torch_helpers
import torch
import os
import random
import numpy as np
from time import time
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.5,
                    help="Learning rate.")
parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5,
                          help="Learning rate decays by this much.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size to use during training.")
parser.add_argument("--enc_embedding_size", default=256, type=int,
                    help="Size encoder embedding layer.")
parser.add_argument("--dec_embedding_size", default=256, type=int,
                    help="Size of decoder embedding layer.")
parser.add_argument("--hidden_units", default=256, type=int,
                    help="Size of S2S encoder and decoder layers.")
parser.add_argument("--slu_hidden_units", default=128, type=int,
                    help="Size of slu layer.")
parser.add_argument("--dm_hidden_units", default=256, type=int,
                    help="Size of dialogue model layers.")
parser.add_argument("--seed", default=923512, type=int,
                    help="Random seed")

parser.add_argument("--s2s_model_dir", default="models_256_512/s2s_model_next_0/",
                    type=str, help="Seq2seq model directory.")
parser.add_argument("--s2s_test_dir", default="test/",
                    type=str, help="Seq2seq testing data directory.")
parser.add_argument("--turn_embeddings_dir", default="turn_emb/",
                    type=str, help="Turn embeddings directory.")
parser.add_argument("--dm_model_dir", default="dm_model_1/",
                    type=str, help="Dialogue Model model directory.")
parser.add_argument("--dm_test_dir", default="dm_test/",
                    type=str, help="Dialogue Model testing data directory.")
parser.add_argument("--dm_learning_rate", default=0.5,
                    type=float, help="Dialogue model learning rate")
parser.add_argument("--data_dir", default="new_data/",
                    type=str, help="Data directory.")

parser.add_argument("--prev_t", default=True,
                    type=str2bool, help="Set to True to predict previous turn.")
parser.add_argument("--next_t", default=True,
                    type=str2bool, help="Set True to predict next turn.")
parser.add_argument("--slu", default=True,
                    type=str2bool, help="Set True to predict slu concept tags.")
parser.add_argument("--s2s_train", default=True,
                    type=str2bool, help="Set to True to train a seq2seq model and compute turn embeddings")
parser.add_argument("--s2s_test", default=True,
                    type=str2bool, help="Set to True to test the s2s model.")
parser.add_argument("--dm_train", default=False,
                    type=str2bool, help="Set to True to train a dialogue model.")
parser.add_argument("--dm_test", default=False,
                    type=str2bool, help="Set to True to test a dialogue model.")
parser.add_argument("--log_file", default="log.txt",type=str,
                    help="File to log all the results.")
parser.add_argument("--decode", default=False,type=str2bool,
                    help="Set True to decode a sentence from stdin.")
parser.add_argument("--quick_test", default=True,
                    type=str2bool, help="Sets max_epochs to 1, in order to test the network quickly.")
parser.add_argument("--use_cuda", default=True,type=str2bool,
                    help="Set true to use cuda if possible.")

FLAGS = parser.parse_args()
class input_data():
    def __init__(self) :
        self.enc_word2idx, self.enc_idx2word = torch_helpers.read_dictionary(FLAGS.data_dir + "user_vocabulary.txt")
        self.dec_word2idx, self.dec_idx2word = torch_helpers.read_dictionary(FLAGS.data_dir + "machine_vocabulary.txt")
        # training data
        self.train_user_data = torch_helpers.read_data(FLAGS.data_dir + "train_user_turns.txt", self.enc_word2idx)
        self.train_prev_system_data = torch_helpers.read_data(FLAGS.data_dir + "train_prev_system_turns.txt", self.dec_word2idx)
        self.train_next_system_data = torch_helpers.read_data(FLAGS.data_dir + "train_next_system_turns.txt", self.dec_word2idx)
        # development data
        self.dev_user_data = torch_helpers.read_data(FLAGS.data_dir + "dev_user_turns.txt", self.enc_word2idx)  # inputs
        self.dev_prev_system_data = torch_helpers.read_data(FLAGS.data_dir + "dev_prev_system_turns.txt", self.dec_word2idx)  # targets
        self.dev_next_system_data = torch_helpers.read_data(FLAGS.data_dir + "dev_next_system_turns.txt", self.dec_word2idx)  # targets

        # read test data
        self.test_user_data = torch_helpers.read_data(FLAGS.data_dir + "test_user_turns.txt", self.enc_word2idx)
        self.test_prev_system_data = torch_helpers.read_data(FLAGS.data_dir + "test_prev_system_turns.txt", self.dec_word2idx)
        self.test_next_system_data = torch_helpers.read_data(FLAGS.data_dir + "test_next_system_turns.txt", self.dec_word2idx)

        # slu data
        self.slu_word2idx, self.slu_idx2word = torch_helpers.read_dictionary(FLAGS.data_dir + "slu_vocabulary.txt")
        self.train_slu_data = torch_helpers.read_data(FLAGS.data_dir + "train_slu_turns.txt", self.slu_word2idx)
        self.dev_slu_data = torch_helpers.read_data(FLAGS.data_dir + "dev_slu_turns.txt", self.slu_word2idx)
        self.test_slu_data = torch_helpers.read_data(FLAGS.data_dir + "test_slu_turns.txt", self.slu_word2idx)

        self.s2s_model_path = FLAGS.s2s_model_dir
        self.s2s_test_path = self.s2s_model_path + FLAGS.s2s_test_dir
        self.turn_embeddings_path = self.s2s_model_path + FLAGS.turn_embeddings_dir
        self.dm_model_path = self.s2s_model_path + FLAGS.dm_model_dir
        self.dm_test_path = self.dm_model_path + FLAGS.dm_test_dir
        self.PAD = 0
        self.GO = 1
        self.EOS = 2
        self.enc_vocab_size = len(self.enc_word2idx)
        self.dec_vocab_size = len(self.dec_word2idx)
        self.slu_vocab_size = len(self.slu_word2idx)
        self.enc_embedding_size = FLAGS.enc_embedding_size
        self.dec_embedding_size = FLAGS.dec_embedding_size
        self.hidden_units = FLAGS.hidden_units
        self.slu_hidden_units = FLAGS.slu_hidden_units
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
        self.prev_t = FLAGS.prev_t  # predict previous system turn
        self.next_t = FLAGS.next_t  # predict next system turn
        self.slu = FLAGS.slu  # predict concep tag
        # dialog model parameters
        self.turn_embedding_size = 2 * self.hidden_units
        self.dm_hidden_units = FLAGS.dm_hidden_units
        self.dm_learning_rate = FLAGS.dm_learning_rate
        self.log_file = FLAGS.log_file
        self.seed = FLAGS.seed
        self.quick_test = FLAGS.quick_test
        self.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
        self.data_dir = FLAGS.data_dir


g = input_data()


if FLAGS.s2s_train:
    torch_seq2seq.train(g)
if FLAGS.s2s_test:
    torch_seq2seq.test(g)
# if not os.path.exists(g.s2s_model_path):
#     os.makedirs(g.s2s_model_path)
# start = time()
# #if decode is true then all the others must be false
# if (not FLAGS.decode) and FLAGS.s2s_train:
#     with open(g.log_file, "a") as f:
#         f.write("S2S Model\n")
#     seq2seq.train(g)
#     print("Training complete.")
#     turn_embeddings.embed(g)
#     print("Turn Embeddings complete.")
#
# if (not FLAGS.decode) and FLAGS.s2s_test:
#     seq2seq_test.test(g)
#     print("Testing complete.")
#
# if (not FLAGS.decode) and FLAGS.dm_train:
#     with open(g.log_file, "a") as f:
#         f.write("Dialogue Model\n")
#     dialogue_model.train(g)
#     print("DM Training complete.")
#
# if (not FLAGS.decode) and FLAGS.dm_test:
#     dm_test.test(g)
#
# end = time()
# with open(g.log_file, "a") as f:
#     f.write("Elapsed time: {:.2f}\n".format(end-start))
#
