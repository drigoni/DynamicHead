#!/usr/bin/env python
"""
Created on 20/12/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: 
"""

import json


def extract_COCO_concepts(ewiser_path, top_pred = 1):
    # reading data from file
    with open(ewiser_path, 'r') as f:
        ewiser_data = json.load(f)
    # extracting synsets
    all_synsets = []
    all_sentences = []
    for sentence in ewiser_data:
        all_sentences.append(sentence['sentence'])
        ewiser = sentence['ewiser']
        for part in ewiser:
            n = part['n_synsets']
            offsets = part['offsets']
            synsets = part['synsets']
            # select just one synset among the top10 and filter them to be just noun
            if n > 0:
                synsets_filtered = [s for s in synsets if '.n.' in s]
                if len(synsets_filtered) > 0:
                    best_synsets = synsets_filtered[:top_pred]
                    all_synsets.extend(best_synsets)
    # make unique list
    all_synsets = list(set(all_synsets))
    if len(all_synsets) == 0:
        print("No noun synset for {}.".format(ewiser_path))
    return all_synsets, all_sentences

