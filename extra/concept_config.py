from detectron2.config import CfgNode as CN


def add_concept_config(cfg):

    # extra configs for concepts and deepsets
    cfg.CONCEPT = CN()
    cfg.CONCEPT.APPLY_CONDITION = True
    cfg.CONCEPT.APPLY_FILTER = True
    cfg.CONCEPT.FILE = './concept/coco_to_synset.json'
    cfg.CONCEPT.VOCAB = './concept/vocab.json'
    cfg.CONCEPT.DEPTH = 3
    cfg.CONCEPT.UNIQUE = True
    cfg.CONCEPT.ONLY_NAME = True

    cfg.DEEPSETS = CN()
    cfg.DEEPSETS.EMB = 'wordnet'   # 'random' # 'wordnet'
    cfg.DEEPSETS.FILE = './concept/wn30_holE_500_150_0.1_0.2_embeddings.pickle'
    cfg.DEEPSETS.EMB_DIM = 150
    cfg.DEEPSETS.FREEZE = True
    cfg.DEEPSETS.MLP1_LAYERS = 2    # 3
    cfg.DEEPSETS.MLP2_LAYERS = 1    # 2
    cfg.DEEPSETS.AGGREGATE = 'sum'  # 'mean'
    cfg.DEEPSETS.OUTPUT_DIM = 256   # 156 with groupnorm 29     # 256 with groupnorm 32