import re
import json

with open("./concept/vg_to_synset.json", "r") as f:
    data = json.load(f)

for k, v in data.items():
    v['coco_cat_id'] -= 1


with open("./concept/vg_to_synset_0.json", "w") as f:
    json.dump(data, f, indent=2)







# MetadataCatalog.get(self.dataset_name[0][0]).thing_dataset_id_to_contiguous_id retrieve a papping dictionary from the notation in the dataset to the interval [0, n_class-1]
# So, the concepts file should use the same ids reported in the dataset .json file, otherwise there would be errors