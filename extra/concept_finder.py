#!/usr/bin/env python
"""
Created on 18/11/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: 
"""

from detectron2.config import CfgNode as CN
import json
import nltk
from nltk.corpus import wordnet as wn
import os
import copy


class ConceptFinder:

    def __init__(self, coco2synset_file='./concept/coco_to_synset.json'):
        print("Load concept for each category. ")
        # load concepts
        with open(coco2synset_file, 'r') as file:
            self.tmp_coco2synset = json.load(file)

        self.coco2synset = dict()
        for category in self.tmp_coco2synset:
            self.tmp_coco2synset[category]['category'] = category
            ids = self.tmp_coco2synset[category]['coco_cat_id']
            self.coco2synset[ids] = self.tmp_coco2synset[category]

    @staticmethod
    def find_descendants(entity, jump, depth=2):
        if jump == 0:
            descendants = [(entity, 0)]
        else:
            descendants = []
        childrens = entity.hyponyms()  # hypernyms for fathers
        descendants = descendants + [(i, jump + 1) for i in childrens]
        if len(childrens) > 0 and jump <= depth:
            for child in childrens:
                new_descendants = ConceptFinder.find_descendants(child, jump + 1)
                descendants = descendants + new_descendants
            return descendants
        else:
            return []

    def extend_descendants(self, depth=2, unique=True, only_name=True):
        coco2synset = copy.deepcopy(self.coco2synset)
        for ids, category in coco2synset.items():
            # find descendant
            tmp_descendants = ConceptFinder.find_descendants(wn.synset(category['synset']), 0, depth)
            tmp_descendants = [(synset.name(), jump) for synset, jump in tmp_descendants]
            # if we need an unique set as synsets.
            # NOTE: if synsets are the same but has different levels, we keep the one with lower level
            if unique:
                tmp_unique_name = []
                tmp_unique_jump = []
                for synset, jump in tmp_descendants:
                    if synset not in tmp_unique_name:
                        tmp_unique_name.append(synset)
                        tmp_unique_jump.append(jump)
                    else:
                        synset_idx = tmp_unique_name.index(synset)
                        if tmp_unique_jump[synset_idx] > jump:
                            tmp_unique_jump[synset_idx] = jump
                category['descendants'] = list(zip(tmp_unique_name, tmp_unique_jump))
            else:
                category['descendants'] = tmp_descendants
            # print(category['name'], category['synset'], category['descendants'])
            if only_name:
                category['descendants'] = [i[0] for i in category['descendants']]
        return coco2synset