from detectron2.config import CfgNode as CN


def add_concept_config(cfg):

    # extra configs for concepts and deepsets
    cfg.CONCEPT = CN()
    cfg.CONCEPT.CONCEPT_FUSION = "cat" # ["cat", "mul", "sum", "zeros"]
    cfg.CONCEPT.APPLY_CONDITION = True
    cfg.CONCEPT.APPLY_CONDITION_FROM_FILE = False
    cfg.CONCEPT.EXTERNAL_CONCEPTS_FOLDER = './datasets/ewiser_concepts_COCO_valid/'
    cfg.CONCEPT.ACTIVATE_CONCEPT_GENERATOR = True
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
    cfg.DEEPSETS.MLP1_LAYERS = [150, 150]
    cfg.DEEPSETS.MLP1_OUTPUT_DIM = 150 
    cfg.DEEPSETS.MLP2_LAYERS = [150]
    cfg.DEEPSETS.OUTPUT_DIM = 256     # 156 with groupnorm 29     # 256 with groupnorm 32
    cfg.DEEPSETS.AGGREGATE = 'sum'  # 'mean'

    # add parameter for new evaluation for the new datasets
    cfg.EVALUATOR_TYPE = 'default' # ['default', postProcessing']