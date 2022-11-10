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

# params
CONCEPT_FILE = './concept/oid_to_synset.json'
CONCEPT_DEPTH = 1
CONCEPT_UNIQUE = True
CONCEPT_ONLY_NAME = True

# load concepts
concept_finder = ConceptFinder(CONCEPT_FILE, depth=CONCEPT_DEPTH, unique=CONCEPT_UNIQUE, only_name=CONCEPT_ONLY_NAME)
coco2synset = concept_finder.coco2synset

all_accepted_concepts = [val_dict['synset'] for k, val_dict in coco2synset.items()]
all_accepted_concepts.extend([v for k, val_dict in coco2synset.items() for v in val_dict['descendants']])
all_accepted_concepts = list(set(all_accepted_concepts))

print("DEPTH VALUE: {} .".format(CONCEPT_DEPTH))
print("Number of conceptsassociated to classes: {} .".format(len(coco2synset.keys())))
print("Number of conceptsassociated to classes: {} .".format(len(all_accepted_concepts)))



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
