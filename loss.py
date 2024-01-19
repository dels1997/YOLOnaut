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

    def forward(
            self, prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
        prediction, target = prediction.to(self.device), target.to(self.device)

        # Reshape into (N, S, S, C + 5 * B) shape, with N corresponding to batch
        # size, S to number of cells along each of the two coordinates,
        # C to the number of classes (one hot encoding) and 5 * B to x, y, w, h
        # and c for each box, respectively. Thus, along that final dimension
        # there are C values representing one hot class encoding, and then
        # 5 values x, y, w, h, c for each of the boxes.
        prediction = \
            prediction.reshape((-1, self.S, self.S, self.C + 5 * self.B))

        # We also reshape the target so that it is of the same shape as
        # prediction, and then later on, using bitmasks, we select out the
        # relevant parts in both of them simultaneously. Here we simply stack
        # the target tensor B - 1 times along the last dimension, so that it
        # has the same shape as prediction.
        for b in range(self.B - 1):
            target = torch.cat([target, target[..., -5:]], dim=-1)

        assert prediction.shape == target.shape

        self.Iobj_i = \
            torch.zeros_like(prediction, dtype=torch.int64, device=self.device)
        nonzero_indices = (target[..., -1] == 1)
        self.Iobj_i[nonzero_indices] = 1

        # Iterate over different boxes and find IOUs between predictions and
        # ground truth values.
        IOUs = []
        for b in range(self.B):
            IOUs.append(self.iou(
                # Include penultimate 4 numbers: x, y, w, h.
                prediction[..., self.C + 5 * b:self.C + 5 * b + 4],
                target[..., self.C + 5 * b:self.C + 5 * b + 4]
            ))
        IOUs = torch.stack(IOUs, dim=-1) # shape (N, S, S, B)

        # First dimension corresponds to examples, last to boxes.
        _, max_IOUs_indices = torch.max(IOUs, dim=-1) # shape (N, S, S)
        max_IOUs_indices = \
            max_IOUs_indices.unsqueeze(dim=-1) # shape (N, S, S, 1)

        # Create a binary mask based on which box is responsible for a certain
        # prediction.
        responsible_shape = list(max_IOUs_indices.shape)
        responsible_shape[-1] = self.C + 5 * self.B
        responsible = \
            torch.ones(responsible_shape, dtype=torch.int64, device=self.device)
        responsible[..., :self.C] = 1
        for b in range(self.B):
            responsible[..., self.C + 5 * b:self.C + 5 * b + 5] = \
                (max_IOUs_indices == b)
        assert responsible.shape == \
            (list(responsible.shape)[0], self.S, self.S, self.C + 5 * self.B)

        # Create a binary mask that takes into account only the cells in which
        # there exist objects and only those boxes that are responsible for
        # them.
        self.Iobj_ij = self.Iobj_i * responsible
        self.Inoobj_ij = 1 - self.Iobj_ij

        # First term in the loss function from original paper, corresponding to
        # square values of differences of x and y coordinates. Iobj_ij is used
        # to select out: the cells i in which there exist predictions; the boxes
        # j which are responsible for particular predictions.
        xy_loss = self.calculate_xy_loss(prediction, target)

        # Second term in the loss function from original paper, corresponding to
        # square values of differences of square roots of widths and heights w
        # and h of boxes. As before, Iobj_ij is used to select relevant terms
        # out.
        wh_loss = self.calculate_wh_loss(prediction, target)

        coordinate_loss = xy_loss + wh_loss

        # Third term in the loss function, corresponding to prediction
        # confidence, penalizing based on Iobj_ij, i.e. in places in which there
        # exist predictions and for boxes that are responsible for those
        # predictions. 
        obj_loss = self.calculate_obj_loss(prediction, target)

        # Fourth term in loss function, corresponding to prediction confidence,
        # penalizing based on Inoobj_ij, i.e. in places complementary to those
        # determined by Iobj_ij.
        noobj_loss = self.calculate_noobj_loss(prediction)

        object_loss = obj_loss + noobj_loss

        # Fifth term in the loss function, corresponding to class of detected
        # object, penalizing based on Iobj_i, i.e. in cells in which there
        # exists an object.
        class_loss = self.calculate_class_loss(prediction, target)

        loss = coordinate_loss + object_loss + class_loss

        return loss