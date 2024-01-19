model_parameters = {
    "version": 1,
    "C": 20,
    "S": 7,
    "B": 2,
    "classes": [
      'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
      'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
      'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
    ],
    "image_width": 448,
    "image_height": 448
}

CLASSES_DICT = \
  {value: index for index, value in enumerate(model_parameters["classes"])}
CLASSES_DICT_INV = \
  {index: value for index, value in enumerate(model_parameters["classes"])}

training_parameters = {
    "batch_size": 4,
    "learning_rate": 2e-5,
    "num_epochs": 500,
    "image_directory": '???',
    "target_directory": '???',
    "best_model_checkpoint_path": '???'
  }

EPSILON = 1e-7
THRESHOLD = 0.8
LARGE_VALUE = 1e10
