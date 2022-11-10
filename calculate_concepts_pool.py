import os
import sys
import itertools
import logging
import time
# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from extra import ConceptFinder
import nltk
from nltk.corpus import wordnet as wn
import json
import argparse
import collections

current_dir = os.path.dirname(os.path.abspath(__file__))


def count_pool_dimension(CONCEPT_FILE, CONCEPT_DEPTH, CONCEPT_UNIQUE, CONCEPT_ONLY_NAME=True):
    # load concepts
    concept_finder = ConceptFinder(CONCEPT_FILE, depth=CONCEPT_DEPTH, unique=CONCEPT_UNIQUE, only_name=CONCEPT_ONLY_NAME)
    coco2synset = concept_finder.coco2synset

    all_accepted_concepts = [val_dict['synset'] for k, val_dict in coco2synset.items()]
    all_accepted_concepts.extend([v for k, val_dict in coco2synset.items() for v in val_dict['descendants']])
    all_accepted_concepts = list(set(all_accepted_concepts))

    print("DEPTH VALUE: {} .".format(CONCEPT_DEPTH))
    print("Number of conceptsassociated to classes: {} .".format(len(coco2synset.keys())))
    print("Number of conceptsassociated to classes: {} .".format(len(all_accepted_concepts)))


def print_freq_per_class(CONCEPT_FILE, CONCEPT_DEPTH, CONCEPT_UNIQUE, OUTPUT_FOLDER, CONCEPT_ONLY_NAME=True):
    # load concepts
    concept_finder = ConceptFinder(CONCEPT_FILE, depth=CONCEPT_DEPTH, unique=CONCEPT_UNIQUE, only_name=CONCEPT_ONLY_NAME)
    coco2synset = concept_finder.coco2synset

    all_accepted_concepts = dict()
    for k, val_dict in coco2synset.items():
        tmp_list = [val_dict['synset']]
        tmp_list.extend(val_dict['descendants'])
        all_accepted_concepts[val_dict['category']] = list(set(tmp_list))
    
    if CONCEPT_FILE.split('/')[-1] == 'coco_to_synset.json':
        data_name = "COCO"
    elif CONCEPT_FILE.split('/')[-1] == 'vg_to_synset.json':
        data_name = "VG"
    else:
        data_name = "OID"
    output_file = "{}{}_number_of_concepts_depth_{}.json".format(OUTPUT_FOLDER, data_name, CONCEPT_DEPTH)

    with open(output_file, 'w') as f:
        json.dump(all_accepted_concepts, f, indent=4)
    print("File saved in {} .".format(output_file))



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
                        help='Levels to consider in the knowledge graph. If add_key=False, then args.level just put the list of descendants but do not affect the dataset.',
                        default=3,
                        type=int)
    parser.add_argument('--coco2concepts', dest='coco2concepts',
                        help='File regarding the links from coco classes to concepts.',
                        default="./concept/coco_to_synset.json",
                        type=str)
    parser.add_argument('--unique', dest='unique',
                        help='Generating considering unique property.',
                        default='True',
                        type=lambda x: True if x.lower() == 'true' else False)
    parser.add_argument('--output', dest='output',
                        help='Output folder.',
                        default='./')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # count_pool_dimension(args.coco2concepts, args.level, args.unique)
    print_freq_per_class(args.coco2concepts, args.level, args.unique, args.output)


# COCO
# DEPTH VALUE: 0 .
# Number of conceptsassociated to classes: 80 .
# Number of conceptsassociated to classes: 79 .
# DEPTH VALUE: 1 .
# Number of conceptsassociated to classes: 80 .
# Number of conceptsassociated to classes: 954 .
# DEPTH VALUE: 2 .
# Number of conceptsassociated to classes: 80 .
# Number of conceptsassociated to classes: 2586 .
# DEPTH VALUE: 3 .
# Number of conceptsassociated to classes: 80 .
# Number of conceptsassociated to classes: 5054 .
# DEPTH VALUE: 4 .
# Number of conceptsassociated to classes: 80 .
# Number of conceptsassociated to classes: 7274 .
# DEPTH VALUE: 5 .
# Number of conceptsassociated to classes: 80 .
# Number of conceptsassociated to classes: 8382 .
# DEPTH VALUE: 6 .
# Number of conceptsassociated to classes: 80 .
# Number of conceptsassociated to classes: 8773 .


# VG
# DEPTH VALUE: 0 .
# Number of conceptsassociated to classes: 1600 .
# Number of conceptsassociated to classes: 1203 .
# DEPTH VALUE: 1 .
# Number of conceptsassociated to classes: 1600 .
# Number of conceptsassociated to classes: 8683 .
# DEPTH VALUE: 2 .
# Number of conceptsassociated to classes: 1600 .
# Number of conceptsassociated to classes: 16243 .
# DEPTH VALUE: 3 .
# Number of conceptsassociated to classes: 1600 .
# Number of conceptsassociated to classes: 22691 .
# DEPTH VALUE: 4 .
# Number of conceptsassociated to classes: 1600 .
# Number of conceptsassociated to classes: 27926 .
# DEPTH VALUE: 5 .
# Number of conceptsassociated to classes: 1600 .
# Number of conceptsassociated to classes: 35292 .
# DEPTH VALUE: 6 .
# Number of conceptsassociated to classes: 1600 .
# Number of conceptsassociated to classes: 39155 .



# OID
# DEPTH VALUE: 0 .
# Number of conceptsassociated to classes: 601 .
# Number of conceptsassociated to classes: 588 . 
# DEPTH VALUE: 1 .
# Number of conceptsassociated to classes: 601 .
# Number of conceptsassociated to classes: 4223 .
# DEPTH VALUE: 2 .
# Number of conceptsassociated to classes: 601 .
# Number of conceptsassociated to classes: 8749 .
# DEPTH VALUE: 3 .
# Number of conceptsassociated to classes: 601 .
# Number of conceptsassociated to classes: 13157 .
# DEPTH VALUE: 4 .
# Number of conceptsassociated to classes: 601 .
# Number of conceptsassociated to classes: 16505 .
# DEPTH VALUE: 5 .
# Number of conceptsassociated to classes: 601 .
# Number of conceptsassociated to classes: 18218 .
# DEPTH VALUE: 6 .
# Number of conceptsassociated to classes: 601 .
# Number of conceptsassociated to classes: 18989 .
