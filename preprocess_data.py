import os
import config
# TODO: Use pybboxes package to check transformations in detail.

CLASSES = config.model_parameters["classes"]
C = config.model_parameters["C"]
assert(C == len(CLASSES))

S = config.model_parameters["S"]

CLASSES_DICT = config.CLASSES_DICT
CLASSES_DICT_INV = config.CLASSES_DICT_INV

ROOT_DATASET_PATH = config.ROOT_DATASET_PATH

XML_ANNOTATIONS_PATH = os.path.join(ROOT_DATASET_PATH, 'Annotations')
TENSOR_ANNOTATIONS_PATH = \
    XML_ANNOTATIONS_PATH.replace('Annotations', 'Annotations_tensor')

from utils.data_utils import read_save_xml_to_tensor_notation

# Read the original files in XML notation and convert them to torch tensors to
# be able to directly pass them into the loss function, and save to .pt files.
read_save_xml_to_tensor_notation(
    XML_ANNOTATIONS_PATH,
    TENSOR_ANNOTATIONS_PATH,
    S,
    C,
    CLASSES,
    CLASSES_DICT,
    CLASSES_DICT_INV
)