# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modified for experiments in https://arxiv.org/abs/2007.00145

"""Class for text data."""
import string
import numpy as np
import torch
import copy
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from .TransformerModel import EncoderLayer, Encoder, \
    MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding


class SimpleVocab(object):

    def __init__(self):
        super(SimpleVocab, self).__init__()
        self.word2id = {}
        self.wordcount = {}
        self.word2id['<UNK>'] = 0
        self.wordcount['<UNK>'] = 9e9

    def tokenize_text(self, text):
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = str(text).lower()
        try:
            tokens = text.translate(
                str.maketrans('', '', string.punctuation)).strip().split()
        except AttributeError:
            tokens = text.translate(None, string.punctuation).strip().split()
        return tokens

    def add_text_to_vocab(self, text):
        tokens = self.tokenize_text(text)
        for token in tokens:
            if token not in self.word2id:
                self.word2id[token] = len(self.word2id)
                self.wordcount[token] = 0
            self.wordcount[token] += 1

    def threshold_rare_words(self, wordcount_threshold=5):
        for w in self.word2id:
            if self.wordcount[w] < wordcount_threshold:
                self.word2id[w] = 0

    def encode_text(self, text):
        tokens = self.tokenize_text(text)
        x = [self.word2id.get(t, 0) for t in tokens]
        return x

    def get_size(self):
        return len(self.word2id)


class EmbeddingModel(torch.nn.Module):

    def __init__(self,
                 texts_to_build_vocab,
                 word_embed_dim=512):

        super(EmbeddingModel, self).__init__()

        self.vocab = SimpleVocab()
        for text in texts_to_build_vocab:
            self.vocab.add_text_to_vocab(text)
        vocab_size = self.vocab.get_size()

        self.word_embed_dim = word_embed_dim
        self.embedding_layer = torch.nn.Embedding(vocab_size,
                                                  word_embed_dim)

    def forward(self, input_list):
        """ input x: list of strings"""
        if type(input_list) is list:
            if type(input_list[0]) is str:
                input_list = [self.vocab.encode_text(text)
                              for text in input_list]

        assert type(input_list) is list
        assert type(input_list[0]) is list
        assert type(input_list[0][0]) is int
        return self.forward_encoded_texts(input_list)

    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(tt) for tt in texts]
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for ii in range(len(texts)):
            itexts[:lengths[ii], ii] = torch.tensor(texts[ii])

        # embed words
        itexts = torch.autograd.Variable(itexts).cuda()
        etexts = self.embedding_layer(itexts)

        text_mask = torch.zeros((len(texts), np.max(lengths))).long()
        for ii in range(len(texts)):
            text_mask[ii, :lengths[ii]] = 1

        return etexts.transpose(0, 1), text_mask.cuda()

    def prepend_vocab(self, texts_to_build_vocab):
        """Stores current vocab in self.temp_vocab for future consolidation.
        Current embeddings are lost.
        After calling this method,
        the object is suitable for loading embedding weights,
        after which consolidate_vocab should be called."""
        self.temp_vocab = self.vocab

        self.vocab = SimpleVocab()
        for text in texts_to_build_vocab:
            self.vocab.add_text_to_vocab(text)
        vocab_size = self.vocab.get_size()
        self.embedding_layer = torch.nn.Embedding(
            vocab_size, self.embedding_layer.weight.shape[-1])

    def consolidate_vocab(self):
        old_vocab_size = self.vocab.get_size()
        for text in self.temp_vocab.word2id.keys():
            self.vocab.add_text_to_vocab(text)
        vocab_increase = self.vocab.get_size() - old_vocab_size
        new_embeddings = torch.nn.Embedding(
            vocab_increase, self.embedding_layer.weight.shape[-1])
        all_embeddings_matrix = torch.cat(
            [self.embedding_layer.weight, new_embeddings.weight], dim=0)
        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            all_embeddings_matrix, freeze=False)
        self.temp_vocab = None


class SimpleTransformerEncoderModel(EmbeddingModel):

    def __init__(self,
                 texts_to_build_vocab,
                 word_embed_dim=512,
                 d_model=512,
                 d_ff=512,
                 num_heads=1,
                 num_layers=1,
                 dropout=0.1,
                 max_length=500):

        super().__init__(texts_to_build_vocab, word_embed_dim)

        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.position_encoder = PositionalEncoding(
            d_model, dropout, max_len=max_length)

        self.model = Encoder(
            EncoderLayer(d_model, c(attn), c(ff), dropout),
            num_layers)

        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(tt) for tt in texts]
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for ii in range(len(texts)):
            itexts[:lengths[ii], ii] = torch.tensor(texts[ii])

        # embed words
        itexts = torch.autograd.Variable(itexts).cuda()
        emb_texts = self.embedding_layer(itexts).transpose(0, 1)

        text_mask = torch.zeros((len(texts), np.max(lengths))).long()
        for ii in range(len(texts)):
            text_mask[ii, :lengths[ii]] = 1
        text_mask = text_mask.cuda()

        emb_texts = self.position_encoder(emb_texts)

        output = self.model(emb_texts, text_mask)

        return output, text_mask


class TextLSTMModel(EmbeddingModel):

    def __init__(self,
                 texts_to_build_vocab,
                 word_embed_dim=512,
                 lstm_hidden_dim=512,
                 num_layers=1,
                 dropout=0.1,
                 text_model_sequence_output=False):

        super(TextLSTMModel, self).__init__(texts_to_build_vocab,
                                            word_embed_dim)

        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim,
                                  num_layers=num_layers)
        self.num_layers = num_layers
        self.fc_output = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
        )
        self.text_model_sequence_output = text_model_sequence_output

    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(tt) for tt in texts]
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for ii in range(len(texts)):
            itexts[:lengths[ii], ii] = torch.tensor(texts[ii])

        # embed words
        itexts = torch.autograd.Variable(itexts).cuda()
        etexts = self.embedding_layer(itexts)

        # lstm
        lstm_output, _ = self.forward_lstm_(etexts)

        if self.text_model_sequence_output:
            # generate mask
            text_mask = torch.zeros((len(texts), np.max(lengths))).long()
            for ii in range(len(texts)):
                text_mask[ii, :lengths[ii]] = 1
            output = self.fc_output(torch.transpose(lstm_output, 0, 1))
            return output, text_mask.cuda()
        else:
            # get last output (using length)
            text_features = []
            for ii in range(len(texts)):
                text_features.append(lstm_output[lengths[ii] - 1, ii, :])

            # output
            text_features = torch.stack(text_features)
            text_features = self.fc_output(text_features)

            return text_features

    def forward_lstm_(self, etexts):
        batch_size = etexts.shape[1]
        first_hidden = (torch.zeros(self.num_layers, batch_size,
                                    self.lstm_hidden_dim),
                        torch.zeros(self.num_layers, batch_size,
                                    self.lstm_hidden_dim))
        first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        return lstm_output, last_hidden


class BERTModel(torch.nn.Module):
    def __init__(self, word_embed_dim=512):

        super(BERTModel, self).__init__()

        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model.eval()
        # import logging
        # logging.basicConfig(level=logging.INFO)
        # self.fc_output = torch.nn.Sequential(
        #    torch.nn.Dropout(p=0.1),
        #    torch.nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
        # )

    def forward(self, x):
        """ input x: batch sentence strings"""
        sentences = x
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]
        indexed_tokens = [torch.tensor(
                            self.tokenizer.convert_tokens_to_ids(tok))
                          for tok in tokenized_texts]
        tokens_tensor = pad_sequence(indexed_tokens, batch_first=True)
        tokens_tensor = tokens_tensor.to('cuda')
        attention_masks = []
        for seq in tokens_tensor:
            seq_mask = [float(ii > 0) for ii in seq]
            attention_masks.append(seq_mask)
        attention_masks = torch.tensor(attention_masks)
        attention_masks = attention_masks.to('cuda')

        with torch.no_grad():
            bert_features, _ = self.model(tokens_tensor, token_type_ids=None,
                                          attention_mask=attention_masks,
                                          output_all_encoded_layers=False)
        # text_features = self.fc_output(bert_features)
        return bert_features, attention_masks
