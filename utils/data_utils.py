"""Implement several conversion functions between different forms of data as an
exercise and a double check."""

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import os
import xmltodict
import torch
import torch.utils.data
from PIL import Image

# TODO: Add type annotations.
def encode(_classes, CLASSES):
    """Create a one-hot encoding in matrix form for a string list '_classes'
    using scikit-learn, preserving their order.
    """
    encoder = \
        OneHotEncoder(categories=[CLASSES], sparse_output=False, dtype=int)
    encoding_matrix = encoder.fit_transform([[c] for c in _classes])

    return encoding_matrix.tolist()

def class_name_to_class_index(_class, CLASSES_DICT):
    return CLASSES_DICT[_class]

def class_index_to_class_name(_num, CLASSES_DICT_INV):
    return CLASSES_DICT_INV[_num]

def xml_to_relative_notation(xml_dict, CLASSES_DICT):
    """Convert the original XML notation to relative notation as a list of
    elements of format (class_index, x, y, w, h), where all x, y, w and h are
    with respect to the whole image."""
    xml_dict_annotation = xml_dict['annotation']
    image_width, image_height = \
        int(xml_dict_annotation['size']['width']), \
        int(xml_dict_annotation['size']['height'])

    if type(xml_dict_annotation['object']) != list:
        xml_dict_annotation['object'] = [xml_dict_annotation['object']]

    annotations = []
    for obj in xml_dict_annotation['object']:
        xmin, xmax, ymin, ymax = \
            float(obj['bndbox']['xmin']), float(obj['bndbox']['xmax']), \
            float(obj['bndbox']['ymin']), float(obj['bndbox']['ymax'])

        annotations.append([
            class_name_to_class_index(obj['name'], CLASSES_DICT),
            (xmin + xmax) / 2 / image_width,
            (ymin + ymax) / 2 / image_height,
            (xmax - xmin) / image_width,
            (ymax - ymin) / image_height
        ])

    return annotations

def read_save_xml_to_relative_notation(
        src_directory, dst_directory, CLASSES_DICT):
    """Read files in XML notation, convert them to relative coordinates in
    format (class_number, x, y, w, h), with coordinates being with respect to
    the whole image, and then save them."""
    os.makedirs(dst_directory, exist_ok=True)

    src_filenames = sorted(os.listdir(src_directory))
    for src_filename in tqdm(src_filenames):
        with open(os.path.join(src_directory, src_filename), 'r') as src_file:
            xml_dict = xmltodict.parse(src_file.read())

        dst_filename = src_filename.replace('xml', 'txt')
        with open(os.path.join(dst_directory, dst_filename), 'w+') as dst_file:
            for annotation in xml_to_relative_notation(xml_dict, CLASSES_DICT):
                dst_file.write(
                    ' '.join(str(elem) for elem in annotation) + '\n'
                )

def xml_to_yolo_notation(
        annotation, image_width, image_height, S, CLASSES, CLASSES_DICT_INV):
    """Convert a single XML annotation to YOLO format (cell_x, cell_y,
    class_onehot, x, y, w, h, c), where x and y are with respect to the
    particular cell."""
    class_num, x, y, w, h = annotation
    cell_width, cell_height = image_width / S, image_height / S

    # TODO: Write equivalent, shorter formulae.
    yolo_x = (x * image_width - int(x * S) * cell_width) / cell_width
    yolo_y = (y * image_height - int(y * S) * cell_height) / cell_height
    yolo_annotation = [
        int(x * S), int(y * S),
        *encode([
            class_index_to_class_name(class_num, CLASSES_DICT_INV)
        ], CLASSES)[0],
        yolo_x, yolo_y, w, h, 1
    ]

    return yolo_annotation

def read_save_xml_to_yolo_notation(
        src_directory, dst_directory, S, CLASSES, CLASSES_DICT, CLASSES_DICT_INV
    ):
    """Read files in XML notation, convert them to .txt files in which each line
    is of format (class_number, x, y, w, h), with x and y relative to a
    particular cell and w and h relative to the whole image, as in the original
    YOLO paper."""
    os.makedirs(dst_directory, exist_ok=True)

    src_filenames = sorted(os.listdir(src_directory))
    for src_filename in tqdm(src_filenames):
        with open(os.path.join(src_directory, src_filename), 'r') as src_file:
            xml_dict = xmltodict.parse(src_file.read())

        image_width = float(xml_dict['annotation']['size']['width'])
        image_height = float(xml_dict['annotation']['size']['height'])
        annotations = xml_to_relative_notation(xml_dict, CLASSES_DICT)

        dst_filename = src_filename.replace('xml', 'txt')
        with open(os.path.join(dst_directory, dst_filename), 'w+') as dst_file:
            for annotation in annotations:
                yolo_annotation = xml_to_yolo_notation(
                    annotation, image_width, image_height, S, CLASSES,
                    CLASSES_DICT_INV
                )
                
                dst_file.write(
                    ' '.join(str(elem) for elem in yolo_annotation) + '\n'
                )

def convert_yolo_to_tensor_notation(list_of_boxes, S, C):
    """Converts the YOLO notation of class encoding and relative coordinates to
    actual tensors used in training, each of shape (S, S, C + 5). This is
    convenient to do since then the tensors can be quickly loaded and direcly
    passed to the loss function."""
    tensor_annotation = torch.zeros((S, S, C + 5), dtype=torch.float32)

    for box in list_of_boxes:
        cell_x, cell_y = int(box[0]), int(box[1])

        _tensor = torch.tensor(box[2:])

        tensor_annotation[cell_x, cell_y] = _tensor

    return tensor_annotation

def read_save_yolo_to_tensor_notation(src_directory, dst_directory, S, C):
    """Read YOLO notation from .txt files, convert to tensors and save to .pt
    files."""
    os.makedirs(dst_directory, exist_ok=True)

    src_filenames = sorted(os.listdir(src_directory))
    for src_filename in tqdm(src_filenames):
        dst_filename = src_filename.replace('txt', 'pt')
        yolo_list = []
        with open(os.path.join(src_directory, src_filename), 'r') as file:
            for line in file:
                line = line.rstrip()
                _targ = [float(x) for x in line.split(' ')]
                yolo_list.append(_targ)

        tensor_annotation = convert_yolo_to_tensor_notation(yolo_list, S, C)

        torch.save(tensor_annotation, os.path.join(dst_directory, dst_filename))

def read_save_xml_to_tensor_notation(
        src_directory, dst_directory, S, C,
        CLASSES, CLASSES_DICT, CLASSES_DICT_INV
    ):
    """Read files in XML notation, convert them to .pt files, each containing a
    tensor of shape (S, S, C + 5)."""
    os.makedirs(dst_directory, exist_ok=True)

    src_filenames = sorted(os.listdir(src_directory))
    for src_filename in tqdm(src_filenames):
        with open(os.path.join(src_directory, src_filename), 'r') as src_file:
            xml_dict = xmltodict.parse(src_file.read())

        annotations = xml_to_relative_notation(xml_dict, CLASSES_DICT)

        image_width = float(xml_dict['annotation']['size']['width'])
        image_height = float(xml_dict['annotation']['size']['height'])

        yolo_annotations = []
        for annotation in annotations:
            yolo_annotations.append(xml_to_yolo_notation(
                    annotation, image_width, image_height, S,
                    CLASSES, CLASSES_DICT_INV
                )
            )

        tensor_annotation = \
            convert_yolo_to_tensor_notation(yolo_annotations, S, C)

        dst_filename = src_filename.replace('xml', 'pt')

        torch.save(tensor_annotation, os.path.join(dst_directory, dst_filename))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory, target_directory, transform=None):
        # TODO: Assert lengths.
        self.image_directory = image_directory
        self.target_directory = target_directory
        self.transform = transform
        self.image_files = sorted(os.listdir(image_directory))
        self.target_files = sorted(os.listdir(target_directory))
        assert(len(self.image_files) == len(self.target_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_directory, self.image_files[index])
        target_path = os.path.join(self.target_directory, self.target_files[index])

        image = Image.open(image_path).convert("RGB")
        target = torch.load(target_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, target