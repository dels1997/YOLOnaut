import torch

import config

EPSILON = config.EPSILON
LARGE_VALUE = config.LARGE_VALUE

class YOLOv1Loss(torch.nn.Module):
    def __init__(
            self, image_width, image_height, S, B, C, λcoord=5, λnoobj=0.5
        ):
        super(YOLOv1Loss, self).__init__()

        self.device = \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_width = image_width
        self.image_height = image_height
        self.cell_width = image_width / S
        self.cell_height = image_height / S
        self.λcoord, self.λnoobj = λcoord, λnoobj
        self.S, self.B, self.C = S, B, C