#!/usr/bin/env python
"""
Created on 16/11/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code needed for creating the new concept dataset.
"""
# import nltk
import os
import json
import argparse
import json
import nltk
from nltk.corpus import wordnet as wn

current_dir = os.path.dirname(os.path.abspath(__file__))
nltk.download('wordnet')

def find_descendants(entity, jump, limit=2):
    childrens = entity.hyponyms()  # hypernyms for fathers
    descendants = [(i, jump) for i in childrens]
    if len(childrens) > 0 and jump < limit:
        for child in childrens:
            new_descendants = find_descendants(child, jump+1)
            descendants = descendants + new_descendants
        return descendants
    else:
        return []

def retrieve_synset(word, clean=True):
    if clean:
        word_cleaned = word.split(' ')
        word_cleaned = '_'.join(word_cleaned)
        word = word_cleaned

    synsets = wn.synsets(word)
    synset = synsets[0] if len(synsets) > 0 else []
    return synset


def create_concept_dataset(annotations_file, level=2, unique=True):
    # read coco annotation file
    print("Loading coco annotations in {} .".format(annotations_file))
    with open(annotations_file, 'r') as file:
        annotations = json.load(file)

    # dictionary
    # "info": {...},
    # "licenses": [...],
    # "images": [...],
    # "annotations": [...],
    # "categories": [...], <-- Not in Captions annotations
    # "segment_info": [...] <-- Only in Panoptic annotations

    # write new dataset information
    print("Processing. ")
    annotations['info']['description'] = "COCO 2017 Dataset augmented with concept"
    annotations['info']['contributor'] = "Davide Rigoni"
    annotations['info']['date_created'] = "2021/11/16"

    # associate the right concept to each category
    # NOTE: here we can add a new class if it is needed
    # NOTE2: we use the Ivis-api coco2synsets links (clean) because many categories do not have a clear synset.
    # do it from scratch
    # for category in annotations['categories']:
    #     synset = retrieve_concept(category['name'])
    #     if category['name'] == 'mouse':
    #         synset = wn.synset('mouse.n.04')
    #     elif category['name'] == 'potted plant':
    #         synset = wn.synset('pot.n.04')
    #     print(category['supercategory'], category['name'], synset)
    # loading Ivis-api coco2synsets links
    with open('concept/coco_to_synset.json', 'r') as file:
        coco2synsets = json.load(file)
    # dictionary
    # category_name: {"coco_cat_id": ...,
    #                  "meaning": ...,
    #                  "synset": ...}

    # for each COCO category extracts its synset and all its descendants
    for category in annotations['categories']:
        category['synset'] = coco2synsets[category['name']]['synset']
        # the category "stop sign" is not linked to a valid synset
        if category['name'] == 'stop sign':
            category['synset'] = wn.synset("traffic_signal.n.01").name()
            # it becames "traffic_lights.n.01"
        tmp_descendants = find_descendants(wn.synset(category['synset']), 0)
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

    # save concept_coco annotation file
    output_file_name = annotations_file.split('/')[-1]
    output_file = "datasets/concept_coco/annotations/{}".format(output_file_name)
    with open(output_file, 'w') as file:
        json.dump(annotations, file)
    print("Dataset saved in {} .".format(output_file))


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--root', dest='root',
                        help='Root folder.',
                        default='{}/'.format(current_dir),
                        type=str)
    parser.add_argument('--level', dest='level',
                        help='Levels to consider in the knowledge graph.',
                        default=2,
                        type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    create_concept_dataset('datasets/coco/annotations/instances_train2017.json', args.level)
    create_concept_dataset('datasets/coco/annotations/instances_val2017.json', args.level)

