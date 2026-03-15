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

DATA_ROOT = "/Users/jackwarren430/Documents/Classes/deep learning/CS-4644-Project/data"   # <--- change this to your data root directory

FGVC_TRAIN_IMAGES = os.path.join(DATA_ROOT, "processed/fgvc/images/train")
FGVC_TRAIN_LABELS = os.path.join(DATA_ROOT, "processed/fgvc/labels/train.csv")

FGVC_VAL_IMAGES = os.path.join(DATA_ROOT, "processed/fgvc/images/val")
FGVC_VAL_LABELS = os.path.join(DATA_ROOT, "processed/fgvc/labels/val.csv")

FGVC_TEST_IMAGES = os.path.join(DATA_ROOT, "processed/fgvc/images/test")
FGVC_TEST_LABELS = os.path.join(DATA_ROOT, "processed/fgvc/labels/test.csv")
