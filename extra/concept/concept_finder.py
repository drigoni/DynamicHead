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
import random


class ConceptFinder:
    def __init__(self, coco2synset_file='./concept/coco_to_synset.json', depth=2, unique=True, only_name=True):
        self.depth = depth
        self.unique = unique
        self.only_name = only_name

        print("Load concept for each category. ")
        # load concepts
        with open(coco2synset_file, 'r') as file:
            tmp_coco2synset = json.load(file)

        self._coco2synset = dict()
        for category in tmp_coco2synset:
            tmp_coco2synset[category]['category'] = category
            ids = tmp_coco2synset[category]['coco_cat_id']
            self._coco2synset[ids] = tmp_coco2synset[category]
        
        # update with descendants and ancestors
        self.extend_descendants(depth, unique, only_name)
        self.extend_ancestors(depth, unique, only_name)
        self._extend_synset_tree()
        self._extend_all_synset_tree()

        if False:
            for k, v in self._coco2synset.items():
                category = v['category']
                synset = v['synset']
                # descendants = v['descendants']
                descendant_of_all = [str(d) + '--' + self._coco2synset[d]['category'] + '--' + self._coco2synset[d]['synset'] for d in v['descendant_of_all']]
                same_as = [str(d) + '--' + self._coco2synset[d]['category'] for d in v['same_as']]
                print('==========================')
                print(k, category, synset)
                print(same_as)
                print(descendant_of_all)
            exit(1) # RecursionError: maximum recursion depth exceeded

    @property
    def coco2synset(self):
        return copy.deepcopy(self._coco2synset)

    def _extend_synset_tree(self):
        """
        This function adds for element of self._coco2Synset the fields "descendant_of" and "same_as".
        Given an element i, i['descendant_of'] is a list of ids referring to other categories whose synsets are ancestors of i['synset']
        Given an element i, i['same_as'] is a list of ids referring to other categories whose synsets are the same as i['synset']
        """
        for s1, v1 in self._coco2synset.items():
            # always b sure to the existance of the key
            if 'descendant_of' not in v1.keys():
                v1['descendant_of'] = []
            if 'same_as' not in v1.keys():
                v1['same_as'] = []
            # search for the other synsets
            for s2, v2 in self._coco2synset.items():
                if s1 == s2:
                    continue
                s1_synset_name = v1['synset']
                s2_synset_name = v2['synset']
                if s1_synset_name == s2_synset_name:
                    v1['same_as'].append(s2)
                if s1_synset_name in v2['descendants']:
                    v1['descendant_of'].append(s2)
    
    def _extend_all_synset_tree(self):
        """
        This function adds for element of self._coco2Synset the fields "descendant_of_all".
        Given an element i, i['descendant_of_all'] is a list of ids referring to other categories whose synsets are ancestors of i['synset'], following all the ids in i['descendant_off']
        """
        # recursive function
        def _follow_descendants_of(synset_id):
            descendants = []
            for d in self._coco2synset[synset_id]['descendant_of']:
                descendants.append(d)
                descendants.extend(_follow_descendants_of(d))
            return descendants
        # start
        for k, v in self._coco2synset.items():
            v['descendant_of_all'] = _follow_descendants_of(k)


    @staticmethod
    def sample_categories_and_concepts(categories, coco2synset, type='subset'):
        """
        This function samples categories and concepts.
        :param categories: the list of all the categories by ids.
        :param coco2synset: coco2synset mapping
        :param type: type of sampling.
        :return: a unique list of selected categories and a unique list of concepts
        """
        list_unique_cat = list(set(categories))   # unique list of categories
        if type=='subset':
            # This function samples some unique categories, then sample a concept for each of them, and then select the categories related with the concepts sampled.
            # ------ 
            selected_categories = set()
            selected_concepts = set()
            # samples a set of categories. At least one.
            sampled_cat = random.sample(list_unique_cat, random.randint(1, len(list_unique_cat)))
            # select concepts according to sampled categories
            for cat_idx in sampled_cat:
                current_synset = coco2synset[cat_idx]['synset']
                descendants = coco2synset[cat_idx]['descendants']
                all_choices = descendants + [current_synset]
                sampled_discendent_concept = random.choice(all_choices)
                selected_concepts.add(sampled_discendent_concept)
            # slect all categories, according to concepts
            for concept in selected_concepts:
                for cat_idx in list_unique_cat:
                    current_synset = coco2synset[cat_idx]['synset']
                    descendants = coco2synset[cat_idx]['descendants']
                    descendants_of_all = coco2synset[cat_idx]['descendant_of_all']
                    if concept in descendants + [current_synset]:
                        selected_categories.add(cat_idx)
                        selected_categories.union(set(descendants_of_all))
            # filtering categories
            selected_categories = selected_categories.intersection(set(list_unique_cat))
            return list(selected_categories), list(selected_concepts)
        elif type=='all':
            # This function samples concepts for each unique category
            # ------ 
            selected_concepts = set()
            for cat_idx in list_unique_cat:
                current_synset = coco2synset[cat_idx]['synset']
                descendants = coco2synset[cat_idx]['descendants']
                all_choices = descendants + [current_synset]
                sampled_discendent_concept = random.choice(all_choices)
                selected_concepts.add(sampled_discendent_concept)
            return list_unique_cat, list(selected_concepts)
        elif type=='subset_old':
            # The one to use
            # ------ 
            selected_categories = set()
            selected_concepts = []
            sampled_cat = random.sample(list_unique_cat, random.randint(1, len(list_unique_cat)))
            categories_for_annotations = [cat for cat in categories if cat in sampled_cat]
            for cat_idx in categories_for_annotations:
                current_synset = coco2synset[cat_idx]['synset']
                descendants = coco2synset[cat_idx]['descendants']
                all_choices = descendants + [current_synset]
                sampled_discendent_concept = random.choice(all_choices)
                selected_concepts.append(sampled_discendent_concept)
            # slect all categories, according to concepts
            for concept in selected_concepts:
                for cat_idx in list_unique_cat:
                    current_synset = coco2synset[cat_idx]['synset']
                    descendants = coco2synset[cat_idx]['descendants']
                    descendants_of_all = coco2synset[cat_idx]['descendant_of_all']
                    if concept in descendants + [current_synset]:
                        selected_categories.add(cat_idx)
                        selected_categories.union(set(descendants_of_all))
            return list(selected_categories), selected_concepts
        elif type=='subset_old_v2':     # the first tried
            # This function samples some unique categories, then sample a concept for each annotation belonging to the category sampled
            # ------ 
            selected_concepts = []
            sampled_cat = random.sample(list_unique_cat, random.randint(1, len(list_unique_cat)))
            categories_for_annotations = [cat for cat in categories if cat in sampled_cat]
            for cat_idx in categories_for_annotations:
                current_synset = coco2synset[cat_idx]['synset']
                descendants = coco2synset[cat_idx]['descendants']
                all_choices = descendants + [current_synset]
                sampled_discendent_concept = random.choice(all_choices)
                selected_concepts.append(sampled_discendent_concept)
            return sampled_cat, selected_concepts
        elif type=='all_old':   # the first tried
            # This function samples for each annotation a concept.
            # ------ 
            selected_concepts = []
            for cat_idx in categories:
                current_synset = coco2synset[cat_idx]['synset']
                descendants = coco2synset[cat_idx]['descendants']
                all_choices = descendants + [current_synset]
                sampled_discendent_concept = random.choice(all_choices)
                selected_concepts.append(sampled_discendent_concept)
            return categories, selected_concepts
        elif type=='query-intent-SLD':
            # This function is to be consistent with: https://arxiv.org/abs/2106.10258
            # ------ 
            selected_categories = set()
            selected_concepts = set()
            # samples just one label
            sampled_cat = [random.choice(list_unique_cat)]
            # select concepts according to sampled categories
            for cat_idx in sampled_cat:
                current_synset = coco2synset[cat_idx]['synset']
                descendants = coco2synset[cat_idx]['descendants']
                all_choices = descendants + [current_synset]
                sampled_discendent_concept = random.choice(all_choices)
                selected_concepts.add(sampled_discendent_concept)
            # slect all categories, according to concepts
            for concept in selected_concepts:
                for cat_idx in list_unique_cat:
                    current_synset = coco2synset[cat_idx]['synset']
                    descendants = coco2synset[cat_idx]['descendants']
                    descendants_of_all = coco2synset[cat_idx]['descendant_of_all']
                    if concept in descendants + [current_synset]:
                        selected_categories.add(cat_idx)
                        selected_categories.union(set(descendants_of_all))
            # filtering categories
            selected_categories = selected_categories.intersection(set(list_unique_cat))
            return list(selected_categories), list(selected_concepts)
        elif type=='query-intent-KLD':
            # This function is to be consistent with: https://arxiv.org/abs/2106.10258
            # ------ 
            selected_categories = set()
            selected_concepts = set()
            # samples just one label
            sampled_cat = [el for el in list_unique_cat if random.random() > 0.5]
            # select concepts according to sampled categories
            for cat_idx in sampled_cat:
                current_synset = coco2synset[cat_idx]['synset']
                descendants = coco2synset[cat_idx]['descendants']
                all_choices = descendants + [current_synset]
                sampled_discendent_concept = random.choice(all_choices)
                selected_concepts.add(sampled_discendent_concept)
            # slect all categories, according to concepts
            for concept in selected_concepts:
                for cat_idx in list_unique_cat:
                    current_synset = coco2synset[cat_idx]['synset']
                    descendants = coco2synset[cat_idx]['descendants']
                    descendants_of_all = coco2synset[cat_idx]['descendant_of_all']
                    if concept in descendants + [current_synset]:
                        selected_categories.add(cat_idx)
                        selected_categories.union(set(descendants_of_all))
            # filtering categories
            selected_categories = selected_categories.intersection(set(list_unique_cat))
            return list(selected_categories), list(selected_concepts)
        else:
            print("Yet to be implemented. ")
            exit(1)

    @staticmethod
    def find_descendants(entity, jump, depth=2):
        """
        This recursive function is needed to find a synset descendats.
        :param jump: recursive values
        :param depth: max distance value
        """
        descendants = [(entity, jump)]
        if jump == depth:
            return descendants
        else:
            childrens = entity.hyponyms()  # hypernyms for fathers
            if len(childrens) > 0:
                # descendants = descendants + [(i, jump) for i in childrens]
                for child in childrens:
                    new_descendants = ConceptFinder.find_descendants(child, jump=jump + 1, depth=depth)
                    descendants = descendants + new_descendants # updated with children if jump+1<=depth 
            return descendants

    def extend_descendants(self, depth=2, unique=True, only_name=True):
        """
        This function adds for element of self._coco2Synset the field "descendats" where all the synset descendants are added.
        :param depth: distance to consider in traveling the graph
        :param unique: True to return a list of unique synsets. If two synsets are equal but just the jump values change, then the lowest value is kept.
        :param unique: True to return only a list of synsets. False to return tuples (synset, jump)
        """
        assert depth >= 0, "Error. Concept depth should be a positive number."
        for ids, category in self._coco2synset.items():
            # find descendant
            tmp_descendants = ConceptFinder.find_descendants(entity=wn.synset(category['synset']), jump=0, depth=depth)
            tmp_descendants = [(synset.name(), jump) for synset, jump in tmp_descendants if jump > 0]   # remove descendants jump=0, meaning the same synset
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

    @staticmethod
    def find_ancestors(entity, jump, depth=2):
        """
        This recursive function is needed to find a synset ancestors.
        :param jump: recursive values
        :param depth: max distance value
        """
        ancestors = [(entity, jump)]
        if jump == depth:
            return ancestors
        else:
            parents = entity.hypernyms()
            if len(parents) > 0:
                for child in parents:
                    new_ancestors = ConceptFinder.find_ancestors(child, jump=jump + 1, depth=depth)
                    ancestors = ancestors + new_ancestors # updated with ancestor if jump+1<=depth 
            return ancestors

    def extend_ancestors(self, depth=2, unique=True, only_name=True):
        """
        This function adds for element of self._coco2Synset the field "ancestors" where all the synset ancestors are added.
        :param depth: distance to consider in traveling the graph
        :param unique: True to return a list of unique synsets. If two synsets are equal but just the jump values change, then the lowest value is kept.
        :param unique: True to return only a list of synsets. False to return tuples (synset, jump)
        """
        for ids, category in self._coco2synset.items():
            # find descendant
            tmp_ancestors = ConceptFinder.find_ancestors(entity=wn.synset(category['synset']), jump=0, depth=depth)
            tmp_ancestors = [(synset.name(), jump) for synset, jump in tmp_ancestors if jump > 0]   # remove ancestors jump=0, meaning the same synset
            # if we need an unique set as synsets.
            # NOTE: if synsets are the same but has different levels, we keep the one with lower level
            if unique:
                tmp_unique_name = []
                tmp_unique_jump = []
                for synset, jump in tmp_ancestors:
                    if synset not in tmp_unique_name:
                        tmp_unique_name.append(synset)
                        tmp_unique_jump.append(jump)
                    else:
                        synset_idx = tmp_unique_name.index(synset)
                        if tmp_unique_jump[synset_idx] > jump:
                            tmp_unique_jump[synset_idx] = jump
                category['ancestors'] = list(zip(tmp_unique_name, tmp_unique_jump))
            else:
                category['ancestors'] = tmp_ancestors
            # print(category['name'], category['synset'], category['ancestors'])
            if only_name:
                category['ancestors'] = [i[0] for i in category['ancestors']]



# class ConceptFinderSynonym(ConceptFinder):
# 
#     def __init__(self, coco2synset_file='./concept/coco_to_synset.json'):
#         super().__init__(
#             coco2synset_file=coco2synset_file
#         )
#         for coco_cat_id in self._coco2synset.keys():
#             category = self._coco2synset[coco_cat_id]['category']
#             self._coco2synset[coco_cat_id]['synset'] = [self._coco2synset[coco_cat_id]['synset']]
#             self._coco2synset[coco_cat_id]['synset'] += [s.name() for s in wn.synsets(category)]
# 
# 
# 
#     @staticmethod
#     def find_descendants(entity, jump, depth=2):
#         return ConceptFinderSynonym.find_descendants(entity, jump, depth)
# 
#     def extend_descendants(self, depth=2, unique=True, only_name=True):
#         coco2synset = self._coco2synset
#         for ids, category in coco2synset.items():
#             tmp_all_descendants = []
#             for synset_name in category['synset']:
#                 synset = wn.synset(synset_name)
#                 # find descendant
#                 tmp = ConceptFinderSynonym.find_descendants(entity=synset, jump=0, depth=depth)
#                 tmp_all_descendants.extend([(synset.name(), jump) for synset, jump in tmp])
#             # if we need an unique set as synsets.
#             # NOTE: if synsets are the same but has different levels, we keep the one with lower level
#             if unique:
#                 tmp_unique_name = []
#                 tmp_unique_jump = []
#                 for synset, jump in tmp_all_descendants:
#                     if synset not in tmp_unique_name:
#                         tmp_unique_name.append(synset)
#                         tmp_unique_jump.append(jump)
#                     else:
#                         synset_idx = tmp_unique_name.index(synset)
#                         if tmp_unique_jump[synset_idx] > jump:
#                             tmp_unique_jump[synset_idx] = jump
#                 category['descendants'] = list(zip(tmp_unique_name, tmp_unique_jump))
#             else:
#                 category['descendants'] = tmp_all_descendants
#             # print(category['name'], category['synset'], category['descendants'])
#             if only_name:
#                 category['descendants'] = [i[0] for i in category['descendants']]
#         return coco2synset
# 
# 
#     @staticmethod
#     def find_ancestors(entity, jump, depth=2):
#         return ConceptFinderSynonym.find_ancestors(entity, jump, depth)
# 
#     def extend_ancestors(self, depth=2, unique=True, only_name=True):
#         coco2synset = self._coco2synset
#         for ids, category in coco2synset.items():
#             tmp_all_ancestors = []
#             for synset_name in category['synset']:
#                 synset = wn.synset(synset_name)
#                 # find descendant
#                 tmp = ConceptFinderSynonym.find_ancestors(entity=synset, jump=0, depth=depth)
#                 tmp_all_ancestors.extend([(synset.name(), jump) for synset, jump in tmp])
#             # if we need an unique set as synsets.
#             # NOTE: if synsets are the same but has different levels, we keep the one with lower level
#             if unique:
#                 tmp_unique_name = []
#                 tmp_unique_jump = []
#                 for synset, jump in tmp_all_ancestors:
#                     if synset not in tmp_unique_name:
#                         tmp_unique_name.append(synset)
#                         tmp_unique_jump.append(jump)
#                     else:
#                         synset_idx = tmp_unique_name.index(synset)
#                         if tmp_unique_jump[synset_idx] > jump:
#                             tmp_unique_jump[synset_idx] = jump
#                 category['ancestors'] = list(zip(tmp_unique_name, tmp_unique_jump))
#             else:
#                 category['ancestors'] = tmp_all_ancestors
#             # print(category['name'], category['synset'], category['ancestors'])
#             if only_name:
#                 category['ancestors'] = [i[0] for i in category['ancestors']]
#         return coco2synset
# 