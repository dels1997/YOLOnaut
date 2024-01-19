import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch

import config

CLASSES_DICT_INV = config.CLASSES_DICT_INV
PAD = config.LABEL_PAD
S, C, B = config.model_parameters['S'], config.model_parameters['C'], \
    config.model_parameters['B']
THRESHOLD = config.THRESHOLD
FONT_SIZE_RATIO_OF_IMAGE_WIDTH = config.FONT_SIZE_RATIO_OF_IMAGE_WIDTH

font_path = config.LABEL_FONT_PATH

def visualize_single_prediction_tensor_notation(
        image_path, target_path, S=S, C=C,THRESHOLD=THRESHOLD,
        color_map_name='hsv_r', font_path=font_path
    ):
    im = Image.open(image_path)
    _, ax = plt.subplots()
    ax.imshow(im)

    # Since we are visualizing a single perdiction, the first dimension is 1.
    target = torch.load(target_path).squeeze(dim=0)

    # Find the number of boxes B from the target tensor.
    B = int((list(target.shape)[-1] - C) / 5)

    boxes = [target[..., C + 5 * b + 4] for b in range(B)]

    # Find the index of the box with the highest confidence.
    box_index = \
        torch.argmax(torch.stack(boxes, dim=-1), dim=-1).unsqueeze(dim=-1)

    # Define color map and font (size). If font path is not set, use default.
    color_map = plt.get_cmap(color_map_name, C)
    font_size = int(FONT_SIZE_RATIO_OF_IMAGE_WIDTH * im.width)
    font = ImageFont.truetype(font_path, font_size) if font_path else None

    # Find the index of start of the five values (x, y, w, h, c) of the box with
    # the highest confidence for each prediction cell.
    box_start_index = C + 5 * box_index

    # Extract the confidence, class index, and box dimensions of the box with
    # the highest confidence for each prediction cell.
    confidence = target.gather(dim=-1, index=box_start_index + 4)
    class_index = torch.argmax(target[..., :C], dim=-1)
    cell_width, cell_height = im.width / S, im.height / S

    # Find prediction cell (x, y) coordinates for each cell.
    cell_x, cell_y = torch.meshgrid(
        torch.arange(S, dtype=target.dtype, device=target.device),
        torch.arange(S, dtype=target.dtype, device=target.device),
        indexing='ij'
    )
    cell_x, cell_y = cell_x.unsqueeze(dim=-1), cell_y.unsqueeze(dim=-1)

    # Find the box dimensions (x1, y1, x2, y2) for each box, while ensuring that
    # they are within the image bounds.
    x_center = cell_width * (
        cell_x + target.gather(dim=-1, index=box_start_index))
    y_center = cell_height * (
        cell_y + target.gather(dim=-1, index=box_start_index + 1))
    w = im.width * target.gather(dim=-1, index=box_start_index + 2)
    h = im.height * target.gather(dim=-1, index=box_start_index + 3)
    x1 = torch.clamp(x_center - w / 2, 0, im.width - 1)
    y1 = torch.clamp(y_center - h / 2, 0, im.height - 1)
    x2 = torch.clamp(x_center + w / 2, 0, im.width - 1)
    y2 = torch.clamp(y_center + h / 2, 0, im.height - 1)

    # Collect (x, y) coordinates of the prediction cells with confidence higher
    # than the threshold and then slice the tensors accordingly.
    valid_detection_index = torch.nonzero(confidence > THRESHOLD)[..., :2]
    num_valid_detections = valid_detection_index.shape[0]
    x1, y1, x2, y2, class_index = [
        tensor[valid_detection_index[:, 0], valid_detection_index[:, 1]] \
        for tensor in [x1, y1, x2, y2, class_index]
    ]

    draw = ImageDraw.Draw(im)