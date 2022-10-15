import nltk
from nltk.corpus import wordnet as wn
from model_code.linker import linker_vg2yago as lk
import re
import model_code.utils as utils
from model_code.parser.EWISER_parser import synsets_to_yago_representation

# download wordnet data
nltk.download('wordnet')

def get_synset(old_link, word):
    synset = wn.synsets(word)
    if len(synset) > 0:
        synset_offset = str(synset[0].offset())
        synset_name = str(synset[0].name())
        # padding as yago with 9 number
        if ".n." in synset_name:
            for _ in range(8 - len(synset_offset)):
                synset_offset = '0' + synset_offset
                # this function add the number 1
            synset_name = synsets_to_yago_representation(synset_name, synset_offset, stamp=False)
            # check = (old_link[-len(synset_offset):] == synset_offset)
            check = (old_link == synset_name)
            res = (old_link, synset_name, check)
            return True, res
        else:
            res = (old_link, None, None)
            return False, res
    else:
        res = (old_link, None, None)
        return False, res

def merge_structures(old_links, new_links):
    final_dict = {}
    for key, old_data in old_links.items():
        new_data = new_links[key]
        if new_data[1] is None:
            final_dict[key] = old_data
        else:
            final_dict[key] = new_data[1]
    return final_dict

def find_wordnet_entities(all_classes):
    res = {}
    for i, v in all_classes.items():
        word_split = re.split('\s', i)
        done, data = get_synset(v, "_".join(word_split))
        if done:
            res[i] = data
        else:
            word_split = re.split(',|\.', i)
            for word in word_split:
                done, data = get_synset(v, word)
                if done:
                    break
            if done:
                res[i] = data
            else:
                res[i] = (v, None, None)
    print("None values: ", len([i[0] for i in res.values()
                                if i[1] is None]))
    print("Diff values: ", len([i[0] for i in res.values()
                                if i[2] is not None and i[2] is False]))
    return res

file_path='./model_code/linker/linker_classes_manual.json'
old_links = lk.get_yago_alignment(file_path)
new_links = find_wordnet_entities(old_links)
results = merge_structures(old_links, new_links)
utils.save_json("./model_code/linker/linker_classes_aut.json", results, 2)