#!/usr/bin/env python
"""
Created on 18/11/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: 
"""

import os
from typing import Dict, List, Callable
import numpy as np
from collections import Counter
import pickle
import json
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchtext.vocab import Vectors, vocab, GloVe


class MLP(nn.Module):
    def __init__(self, in_size: int, out_size: int, hid_sizes: List[int],
                 activation_function: Callable = F.relu,
                 init_function: Callable = nn.init.xavier_normal_):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.activation_function = activation_function
        self.init_function = init_function
        self.layers = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        # definition
        layers = nn.ModuleList([nn.Linear(s[0], s[1]) for (i, s) in enumerate(weight_sizes)])
        # init
        for layer in layers:
            self.init_function(layer.weight)
            nn.init.zeros_(layer.bias)
        return layers

    def forward(self, inputs):
        acts = inputs
        for layer in self.layers:
            hid = layer(acts)
            acts = self.activation_function(hid)
        last_hidden = hid
        return last_hidden


class _DeepSets(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(_DeepSets, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # building the network
        self.mlp1 = MLP(input_size, input_size, [input_size, input_size])
        self.mlp2 = MLP(input_size, output_size, [input_size])

    def forward(self, x, x_mask):
        # x input as [batch, n_concepts, input_size]
        # x_mask input as [batch, n_concepts] with ones where it is not padding.
        x = self.mlp1(x)                    # [batch, n_concepts, input_size]
        x = x * x_mask.unsqueeze(-1)        # [batch, n_concepts, input_size]
        x = torch.sum(x, dim=1)             # [batch, input_size]
        x = self.mlp2(x)                    # [batch, output_size]
        return x


class Word2Vec(Vectors):
    """
    Word2Vec adapter for `torchtext.vocab.Vectors`.
    """
    def cache(self, name, cache, url=None, max_vectors=None):
        file_suffix = f"_{max_vectors}.pt" if max_vectors else ".pt"
        path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix

        if not os.path.isfile(path_pt):
            import gensim.downloader as api
            model = api.load(name)

            self.itos = model.index_to_key
            self.stoi = model.key_to_index
            self.vectors = torch.tensor(model.vectors)
            self.dim = model.vector_size

            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)


class Wordnet(Vectors):
    """
    Wordnet adapter for `torchtext.vocab.Vectors`.
    """
    def __init__(self, name, init_folder, **kwargs):
        self.init_folder = init_folder
        super(Wordnet, self).__init__(name, **kwargs)

    def cache(self, name, cache, url=None, max_vectors=None):
        file_suffix = ".pt"  # es: wn30_holE_500_150_0.1_0.2_embeddings.pickle
        path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix

        if not os.path.isfile(path_pt):
            # loading file .pickle and save all the data in a pt file in .vector_cache folder
            path_file = os.path.join(self.init_folder, name) + ".pickle"
            print("First time loading: {}. Creating cache. ".format(path_file))
            with open(path_file, "rb") as file:
                loaded_embeddings = pickle.load(file)
            loaded_embeddings_mapped = dict()
            for key, emb in loaded_embeddings.items():
                # to avoid lemmas like clay-colored_robin and synsets like adaxial.a.01
                if ".n." in key:
                    loaded_embeddings_mapped[key] = emb
            # get indexes
            index_to_key = {}
            key_to_index = {}
            vectors = list()
            for i, (k, v) in enumerate(loaded_embeddings_mapped.items()):
                index_to_key[i] = k
                key_to_index[k] = i
                vectors.append(v)

            # save iterators
            self.itos = index_to_key
            self.stoi = key_to_index
            self.vectors = torch.tensor(vectors)
            self.dim = self.vectors.shape[-1]
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)


class ConceptNet(torch.nn.Module):
    def __init__(self, cfg):
        super(ConceptNet, self).__init__()
        self.emb_type = cfg.DEEPSETS.EMB
        self.emb_dim = cfg.DEEPSETS.EMB_DIM
        self.output_dim = cfg.DEEPSETS.OUTPUT_DIM
        self.emb_freeze = cfg.DEEPSETS.FREEZE
        self.path_vocab = cfg.CONCEPT.VOCAB
        with open(self.path_vocab, "r") as file:
            self.concept_keys = json.load(file)
        counter = Counter(self.concept_keys)
        self.concept_vocab = vocab(counter) #, specials=['<pad>'], specials_first=True)
        self.concept_emb = ConceptNet.create_embeddings_network(self.emb_type, self.concept_vocab,
                                                                self.emb_dim, self.emb_freeze)
        self.deepset = _DeepSets(self.emb_dim, self.output_dim)

    def forward(self, x, x_mask):
        x = self.concept_emb(x)
        x = self.deepset(x, x_mask)
        return x

    def preprocess_concepts(self, batched_inputs: List[Dict[str, List]]):
        """
        Batch the input concepts with vectors of zeros.
        """
        concepts = [x["concepts"] for x in batched_inputs]      # [b, annotations, concepts]
        # print("CONCEPTS: ", len(concepts), len(concepts[0]))
        batch_size = len(concepts)
        max_n_concepts_for_example = 0
        # tokenize
        concepts_tokenized = self.concept_tokenization(concepts)

        # get info
        for list_of_concepts in concepts_tokenized:
            max_n_concepts_for_example = max(max_n_concepts_for_example, len(list_of_concepts))

        # creating the padding tensor
        results = np.zeros([batch_size, max_n_concepts_for_example])
        mask = np.zeros([batch_size, max_n_concepts_for_example])
        for concept_i, concept_data in enumerate(concepts_tokenized):
            results[concept_i, :len(concept_data)] = concept_data
            mask[concept_i, :len(concept_data)] = [1] * len(concept_data)

        results = torch.tensor(results, dtype=torch.int64)
        mask = torch.tensor(mask, dtype=torch.int32)
        return results, mask

    def concept_tokenization(self, data: List[List[str]]):
        """
        Use the vocabulary to retrieve entity indexes for the concepts.
        """
        results = []
        for batch in data:
            # clean the predictions
            tmp = [self.concept_vocab[concept] for concept in batch]
            results.append(tmp)
        return results

    @staticmethod
    def create_embeddings_network(embedding_type: str, concept_vocab,
                                  emb_size: int, freeze: bool = True):
        """
        This function builds the word embedding network using the defined embeddings.
        :param embedding_type: type of embedding i.e. "glove", "random" or "wordnet"
        :param concept_vocab: concepts vocabulary
        :param emb_size: size of the embeddings
        :param freeze: True for freezing the embeddings, else False
        :return: nn.Embedding object
        """
        print("Loading pre-trained concepts embeddings. ")
        vocab_size = len(concept_vocab) if concept_vocab is not None else 0
        out_of_vocabulary = 0
        if embedding_type == 'random':
            return nn.Embedding(vocab_size + 1, emb_size)
        else:
            embedding_matrix_values = torch.zeros((vocab_size + 1, emb_size),
                                                  requires_grad=(not freeze))  # special char
            pretrained_embeddings = ConceptNet.get_word_embedding(embedding_type, emb_size)
            pretrained_words = pretrained_embeddings.stoi.keys()
            for word_idx in range(vocab_size):
                # print("Loaded {}/{} concepts. ".format(word_idx, vocab_size))
                word = concept_vocab.get_itos()[word_idx]
                if word in pretrained_words:
                    embedding_idx = pretrained_embeddings.stoi[word]
                    embedding_matrix_values[word_idx, :] = pretrained_embeddings.vectors[embedding_idx]
                else:
                    # print("CONCEPT not found: ", word)
                    out_of_vocabulary += 1
                    nn.init.normal_(embedding_matrix_values[word_idx, :])
            embedding_matrix = nn.Embedding(vocab_size, emb_size)
            embedding_matrix.weight = torch.nn.Parameter(embedding_matrix_values)
            embedding_matrix.weight.requires_grad = not freeze
            print('Vocab initialization with {}/{} elements not found. '.format(out_of_vocabulary, vocab_size))
            return embedding_matrix

    @staticmethod
    def get_word_embedding(kind: str, dim: int = 300, **kwargs):
        """
        Return a `torchtext.vocab.Vectors` object, instanced with word embeddings wrt `kind`
        (i.e., fixed, pretrained models).
        :param kind: One of 'glove', 'w2v'
        :param dim: The embedding size
        :param kwargs: Other `torchtext.vocab.Vectors.__init__` parameters
        :return: A `torchtext.vocab.Vectors` instance
        """
        if kind == "glove":
            return GloVe(name='840B', dim=dim, **kwargs)
        if kind == "w2v":
            if dim != 300:
                raise ValueError(f"The specified embedding size ({dim}) is not valid for Word2Vec. Please use `dim=300`.")
            return Word2Vec(name="word2vec-google-news-300", **kwargs)
        if kind == "wordnet":
            if dim != 150:
                raise ValueError(f"The specified embedding size ({dim}) is not valid for Wordnet. Please use `dim=150`.")
            return Wordnet(name="wn30_holE_500_150_0.1_0.2_embeddings",
                           init_folder="./concept/",
                           **kwargs)
        raise ValueError(f"Invalid embedding kind ({kind}), should be in 'glove', 'w2v', 'wordnet'.")
