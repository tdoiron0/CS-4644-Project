import os


'''
    MODELS
'''

MODEL_INTERNVL3_2B = "OpenGVLab/InternVL3-2B-hf"
MODEL_INTERNVL3_8B = "OpenGVLab/InternVL3-8B-hf"
MODEL_INTERNVL3_14B = "OpenGVLab/InternVL3-14B-hf"


'''
    DEVICES
'''

DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"
DEVICE_MPS = "mps"


'''
    DATA PATHS
'''

DATA_ROOT = "/Users/seanhall/Desktop/cs4644 project/data"

FGVC_TRAIN_IMAGES = os.path.join(DATA_ROOT, "processed/fgvc/images/train")
FGVC_TRAIN_LABELS = os.path.join(DATA_ROOT, "train.csv")

FGVC_VAL_IMAGES = os.path.join(DATA_ROOT, "processed/fgvc/images/val")
FGVC_VAL_LABELS = os.path.join(DATA_ROOT, "val.csv")

FGVC_TEST_IMAGES = os.path.join(DATA_ROOT, "processed/fgvc/images/test")
FGVC_TEST_LABELS = os.path.join(DATA_ROOT, "processed/fgvc/labels/test.csv")

WIKI_CORPUS = os.path.join(DATA_ROOT, "processed/wikitext/text_corpus.jsonl")
WIKI_CORPUS_EXPANDED = os.path.join(DATA_ROOT, "processed/wikitext/text_corpus_expanded.jsonl")
