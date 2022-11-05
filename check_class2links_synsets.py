import json
import nltk
from nltk.corpus import wordnet as wn

with open("concept/oid_to_synset.json", "r") as f:
    link = json.load(f)

with open("concept/vocab.json", "r") as f:
    vocab = json.load(f)

for key, val in link.items():
    if val['synset'] not in vocab:
        print("Synset not found: ", val['synset'])

