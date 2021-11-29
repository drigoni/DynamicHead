from detectron2.config import CfgNode as CN


def add_concept_config(cfg):

    # extra configs for concepts and deepsets
    cfg.CONCEPT = CN()
    cfg.CONCEPT.FILE = './concept/coco_to_synset.json'
    cfg.CONCEPT.VOCAB = './concept/vocab.json'
    cfg.CONCEPT.DEPTH = 10
    cfg.CONCEPT.UNIQUE = True
    cfg.CONCEPT.ONLY_NAME = True

    cfg.DEEPSETS = CN()
    cfg.DEEPSETS.EMB = 'wordnet'
    cfg.DEEPSETS.FILE = './concept/wn30_holE_500_150_0.1_0.2_embeddings.pickle'
    cfg.DEEPSETS.EMB_DIM = 150
    cfg.DEEPSETS.FREEZE = True
    cfg.DEEPSETS.OUTPUT_DIM = 150
