# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


"""Class for text data."""
import string
import numpy as np
import torch
from .transformer import EncoderLayer, Encoder, \
    MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding
from tokenizers import ByteLevelBPETokenizer  # huggingface
from transformers import RobertaTokenizer, RobertaModel
import os
from ..config.compat import MAAF_ALIASES


class SimpleVocab:

    def __init__(self, max_tokens=128):
        self.word2id = {}
        self.wordcount = {}
        self.word2id['<UNK>'] = 0
        self.wordcount['<UNK>'] = 9e9
        self.max_tokens = max_tokens

    def tokenize_text(self, text):
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = str(text).lower()
        try:
            tokens = text.translate(
                str.maketrans('', '', string.punctuation)).strip().split()
        except AttributeError:
            tokens = text.translate(None, string.punctuation).strip().split()
        if len(tokens) == 0:
            return ['<UNK>']
        return tokens[:self.max_tokens]

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

    def encode_one_text(self, text):
        tokens = self.tokenize_text(text)
        x = [self.word2id.get(t, 0) for t in tokens]
        return x

    def __call__(self, texts, padding=True, return_tensors="pt",
                 truncation=True):
        assert return_tensors == "pt"
        assert truncation
        assert padding
        if type(texts) is list:
            if type(texts[0]) is str:
                texts = [self.encode_one_text(text) for text in texts]

        assert type(texts) is list
        assert type(texts[0]) is list
        try:
            assert type(texts[0][0]) is int
        except IndexError:
            print(texts)
            raise

        output = {}

        # to tensor
        lengths = [len(tt) for tt in texts]
        output["input_ids"] = torch.zeros((np.max(lengths), len(texts)),
                                          dtype=torch.long)
        for ii in range(len(texts)):
            output["input_ids"][:lengths[ii], ii] = torch.tensor(texts[ii])

        output["attention_mask"] = \
            torch.zeros((len(texts), np.max(lengths)), dtype=torch.long)
        for ii in range(len(texts)):
            output["attention_mask"][ii, :lengths[ii]] = 1

        return output

    def __len__(self):
        return len(self.word2id)

    def get_size(self):
        return len(self.word2id)


class EmbeddingModel(torch.nn.Module):

    def __init__(self,
                 tokenizer,
                 word_embed_dim=512):
        super().__init__()

        self.tokenizer = tokenizer
        self.word_embed_dim = word_embed_dim
        self.embedding_layer = torch.nn.Embedding(
            len(tokenizer), word_embed_dim)

    def forward(self, input_list):
        """input x: list of strings"""
        inputs = self.tokenizer(input_list, padding=True, return_tensors="pt",
                                truncation=True)
        etexts = self.embedding_layer(inputs["input_ids"])
        return etexts.transpose(0, 1), inputs["attention_mask"]

    @property
    def device(self):
        """Only makes sense if all parameters on same device."""
        return next(self.parameters()).device


class SimpleTransformerEncoderModel(EmbeddingModel):

    def __init__(self,
                 tokenizer,
                 word_embed_dim=512,
                 d_model=512,
                 d_ff=512,
                 num_heads=1,
                 num_layers=1,
                 dropout=0.1,
                 max_length=500):

        super().__init__(tokenizer, word_embed_dim)

        attn = MultiHeadedAttention(num_heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.position_encoder = PositionalEncoding(
            d_model, dropout, max_len=max_length)

        self.model = Encoder(
            EncoderLayer(d_model, attn, ff, dropout),
            num_layers)

        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt",
                                truncation=True)
        emb_texts = self.embedding_layer(inputs["input_ids"])
        emb_texts = self.position_encoder(emb_texts)

        output = self.model(emb_texts, inputs["attention_mask"])

        return output, inputs["attention_mask"]


class TextLSTMModel(EmbeddingModel):

    def __init__(self,
                 tokenizer,
                 word_embed_dim=512,
                 lstm_hidden_dim=512,
                 num_layers=1,
                 dropout=0.1,
                 text_model_sequence_output=False,
                 output_relu=False):

        super().__init__(tokenizer, word_embed_dim)

        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim,
                                  num_layers=num_layers)
        self.num_layers = num_layers
        fc_output = [torch.nn.Dropout(p=dropout),
                     torch.nn.Linear(lstm_hidden_dim, lstm_hidden_dim)]
        if output_relu:
            fc_output.append(torch.nn.ReLU())
        self.fc_output = torch.nn.Sequential(*fc_output)
        self.text_model_sequence_output = text_model_sequence_output

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt",
                                truncation=True)
        try:
            inputs = inputs.to(self.device)
        except AttributeError:
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
        emb_texts = self.embedding_layer(inputs["input_ids"])

        # lstm
        lstm_output, _ = self.forward_lstm_(emb_texts)

        if self.text_model_sequence_output:
            output = self.fc_output(lstm_output).transpose(0, 1)
            return output, inputs["attention_mask"]
        else:
            # TODO adapt to new framework with tokenizers
            # get last output (using length)
            lengths, _ = inputs["attention_mask"].cumsum(1).max(1)
            text_features = []
            for ii in range(len(texts)):
                text_features.append(lstm_output[lengths[ii] - 1, ii, :])

            text_features = torch.stack(text_features)
            text_features = self.fc_output(text_features)

            return text_features

    def forward_lstm_(self, etexts):
        batch_size = etexts.shape[1]
        first_hidden = (torch.zeros(self.num_layers, batch_size,
                                    self.lstm_hidden_dim, device=etexts.device),
                        torch.zeros(self.num_layers, batch_size,
                                    self.lstm_hidden_dim, device=etexts.device))
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        return lstm_output, last_hidden


class Roberta(torch.nn.Module):

    def __init__(self, tokenizer, model_path, out_channels=512, dropout=0.1,
                 pretrained=True):
        super().__init__()
        self.tokenizer = tokenizer

        local = (not pretrained) or os.environ.get("LOCAL_FILES_ONLY")
        if model_path is None:
            model_path = "distilroberta-base"
        self.model = RobertaModel.from_pretrained(model_path,
                                                  local_files_only=local)
        if out_channels == -1:
            self.out_layer = torch.nn.Identity()
        else:
            self.out_layer = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(768, out_channels),
            )

    def pretrained_parameters(self):
        scratch = set([param for param in self.out_layer.parameters()])
        all_param = set([param for param in self.parameters()])
        return all_param.difference(scratch)

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt",
                                truncation=True).to(self.model.device)
        out = self.model(**inputs)[0]
        out = self.out_layer(out)
        return out, inputs["attention_mask"]


def build_text_model(texts, cfg):
    if cfg.MODEL.TEXT_MODEL.ARCHITECTURE is None:
        return None

    if cfg.MODEL.TEXT_MODEL.OUTPUT_RELU:
        assert cfg.MODEL.TEXT_MODEL.ARCHITECTURE == "lstm"

    # tokenizer
    if cfg.MODEL.TEXT_MODEL.TOKENIZER == "simple":
        tokenizer = SimpleVocab(max_tokens=cfg.MODEL.TEXT_MODEL.MAX_TOKENS)
        for text in texts:
            tokenizer.add_text_to_vocab(text)

        if cfg.MODEL.TEXT_MODEL.VOCAB_MIN_FREQ > 0:
            print(tokenizer.get_size(), ' total words seen')
            tokenizer.threshold_rare_words(cfg.MODEL.TEXT_MODEL.VOCAB_MIN_FREQ)
            print(len(set(tokenizer.word2id.values())) - 1,
                  ' words seen enough times to keep')
    else:
        if cfg.MODEL.TEXT_MODEL.VOCAB_DATA is not None:
            tokenizer = ByteLevelBPETokenizer(lowercase=True,
                                              add_prefix_space=True)
            print("training tokenizer...")
            tokenizer.train(files=cfg.MODEL.TEXT_MODEL.VOCAB_DATA,
                            vocab_size=cfg.MODEL.TEXT_MODEL.MAX_VOCAB,
                            min_frequency=cfg.MODEL.TEXT_MODEL.VOCAB_MIN_FREQ)
            if not os.path.exists(cfg.MODEL.TEXT_MODEL.TOKENIZER_PATH):
                os.makedirs(cfg.MODEL.TEXT_MODEL.TOKENIZER_PATH)
            tokenizer.save_model(cfg.MODEL.TEXT_MODEL.TOKENIZER_PATH)

        tokenizer = RobertaTokenizer.from_pretrained(
            cfg.MODEL.TEXT_MODEL.TOKENIZER_PATH,
            add_prefix_space=True,
            model_max_length=cfg.MODEL.TEXT_MODEL.MAX_TOKENS)

    # text model
    embed_dim = cfg.MODEL.TEXT_MODEL.EMBED_DIM
    text_model_sequence_output = cfg.MODEL.COMPOSITION in MAAF_ALIASES
    if cfg.MODEL.TEXT_MODEL.ARCHITECTURE == 'embeddings':
        txtmod = EmbeddingModel(tokenizer, word_embed_dim=embed_dim)
        print("Using bare embeddings for text")
    elif cfg.MODEL.TEXT_MODEL.ARCHITECTURE == 'transformer':
        txtmod = SimpleTransformerEncoderModel(
            tokenizer,
            word_embed_dim=embed_dim,
            d_model=embed_dim,
            d_ff=embed_dim,
            num_layers=cfg.MODEL.TEXT_MODEL.NUM_LAYERS,
            dropout=cfg.MODEL.DROPOUT_RATE)
        print("Using transformer model for text")
        text_model_sequence_output = True
    elif cfg.MODEL.TEXT_MODEL.ARCHITECTURE == "roberta":
        if cfg.MODEL.EMBED_DIM == cfg.MODEL.TEXT_MODEL.EMBED_DIM == 768:
            out_channels = -1
        else:
            out_channels = cfg.MODEL.TEXT_MODEL.EMBED_DIM
        txtmod = Roberta(tokenizer, cfg.MODEL.TEXT_MODEL.MODEL_PATH,
                         out_channels=out_channels,
                         dropout=cfg.MODEL.DROPOUT_RATE,
                         pretrained=cfg.MODEL.WEIGHTS is None)
        text_model_sequence_output = True
    else:
        txtmod = TextLSTMModel(
            tokenizer,
            word_embed_dim=embed_dim,
            lstm_hidden_dim=embed_dim,
            text_model_sequence_output=text_model_sequence_output,
            num_layers=cfg.MODEL.TEXT_MODEL.NUM_LAYERS,
            dropout=cfg.MODEL.DROPOUT_RATE,
            output_relu=cfg.MODEL.TEXT_MODEL.OUTPUT_RELU)
    if cfg.MODEL.TEXT_MODEL.FREEZE_WEIGHTS:
        print("Freezing Text model weights")
        for param in txtmod.parameters():
            param.requires_grad = False

    print("vocab size", len(tokenizer))
    return txtmod
