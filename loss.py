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

    def convert_boxes_to_tblr(
            self, cell_x: torch.Tensor, cell_y: torch.Tensor,
            boxes: torch.Tensor
        ) -> torch.Tensor:
        """Converts YOLO coordinates to top-left--bottom-right coordinates,
        making it possible to find overlap between different rectangles (boxes).

        Args:
            cell_x: X coordinates of cells, for all examples in batch and all
                cells in image, of shape (N, S, S, 1).
            cell_y: Y coordinates of cells, for all examples in batch and all
                cells in image, of shape (N, S, S, 1).
            boxes: Rectangles given in YOLO coordinates, for all examples in
                batch and all cells in image, of shape (N, S, S, 4).

        Returns: 
            tblr_boxes: Box rectangles given in top-left--bottom-right
                coordinates, for all examples in batch and all cells in image,
                of shape (N, S, S, 4).
        """
        tblr_boxes = torch.zeros_like(boxes, device=self.device)
        tblr_boxes[..., 0] = self.Iobj_i[..., 4] * (
            cell_x * self.cell_width + \
            boxes[..., 0] * self.cell_width - \
            boxes[..., 2] * self.image_width / 2
        ) # x1 (x coordinate of top-left corner)
        tblr_boxes[..., 1] = self.Iobj_i[..., 4] * (
            cell_y * self.cell_height + \
            boxes[..., 1] * self.cell_height - \
            boxes[..., 3] * self.image_height / 2
        ) # y1 (y coordinate of top-left corner)
        tblr_boxes[..., 2] = self.Iobj_i[..., 4] * (
            cell_x * self.cell_width + \
            boxes[..., 0] * self.cell_width + \
            boxes[..., 2] * self.image_width / 2
        ) # x2 (x coordinate of bottom-right corner)
        tblr_boxes[..., 3] = self.Iobj_i[..., 4] * (
            cell_y * self.cell_height + \
            boxes[..., 1] * self.cell_height + \
            boxes[..., 3] * self.image_height / 2
        ) # y2 (y coordinate of bottom-right corner)

        return tblr_boxes

    def find_tblr_coordinates_of_intersection(
            self, tblr_boxes1: torch.Tensor, tblr_boxes2: torch.Tensor
        ) -> torch.Tensor:
        """Top-left--bottom-right coordinates of intersection are found as
        maxima of top-left corners and minima of right-bottom corners."""
        tblr_edges = torch.zeros(tblr_boxes1.shape, device=self.device)
        tblr_edges[..., 0] = \
            torch.maximum(tblr_boxes1[..., 0], tblr_boxes2[..., 0])
        tblr_edges[..., 1] = \
            torch.maximum(tblr_boxes1[..., 1], tblr_boxes2[..., 1])
        tblr_edges[..., 2] = \
            torch.minimum(tblr_boxes1[..., 2], tblr_boxes2[..., 2])
        tblr_edges[..., 3] = \
            torch.minimum(tblr_boxes1[..., 3], tblr_boxes2[..., 3])

        return tblr_edges

    def iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Returns intersection-over-union of two rectangles given in YOLO
        coordinates by converting them into absolute coordinates in image. Done
        in vectorized way at once for all of the examples in batch and all
        cells in image.

        Args:
            boxes1, boxes2: Rectangles given in YOLO coordinates, for all
                examples in batch and all cells in image, of shape (N, S, S, 4).

        Returns: 
            _iou: IOU values for given box, across all examples and cells, of
                shape (N, S, S, 1).
        """
        # Find row (x) and column (y) of each cell of initial tensors.
        cell_tensor_shape = list(boxes1.shape)
        cell_tensor_shape[-1] = 1

        cell_x = torch.ones(
            cell_tensor_shape, dtype=torch.float32, device=self.device)
        cell_x = torch.einsum(
            'ijkl,j->ijkl', cell_x,
            torch.arange(
                cell_tensor_shape[1], dtype=torch.float32, device=self.device)
        )
        cell_x = cell_x.squeeze()

        cell_y = \
            torch.ones(
                cell_tensor_shape, dtype=torch.float32, device=self.device)
        cell_y = torch.einsum(
            'ijkl,k->ijkl', cell_y,
            torch.arange(
                cell_tensor_shape[2], dtype=torch.float32, device=self.device)
        )
        cell_y = cell_y.squeeze()

        tblr_boxes1 = self.convert_boxes_to_tblr(cell_x, cell_y, boxes1)
        tblr_boxes2 = self.convert_boxes_to_tblr(cell_x, cell_y, boxes2)

        tblr_edges = \
            self.find_tblr_coordinates_of_intersection(tblr_boxes1, tblr_boxes2)

        # Intersection areas are simply found as rectangle areas, while ensuring
        # that in the case of zero overlap, we return zero value instead of a
        # possibly negative one.
        intersection_width = (tblr_edges[..., 2] - tblr_edges[..., 0])
        intersection_height = (tblr_edges[..., 3] - tblr_edges[..., 1])
        intersection = \
            torch.maximum(
                torch.zeros_like(intersection_width), intersection_width
            ) * \
            torch.maximum(
                torch.zeros_like(intersection_height), intersection_height
            )

        # Find the individual box areas.
        box1_area = torch.abs(
            (tblr_boxes1[..., 2] - tblr_boxes1[..., 0]) * \
            (tblr_boxes1[..., 3] - tblr_boxes1[..., 1])
        )
        box2_area = torch.abs(
            (tblr_boxes2[..., 2] - tblr_boxes2[..., 0]) * \
            (tblr_boxes2[..., 3] - tblr_boxes2[..., 1])
        )

        # Area of union of two rectangles is equal to a sum of their areas
        # minus the intersection area. The epsilon value is added so that in
        # case of small areas, there is no issue of numerical instability and
        # extremely large values when we divide by 'union' next.
        union = box1_area + box2_area - intersection + EPSILON

        _iou = intersection / union

        # Add discarding negative values as an additional failsafe. Add
        # discarding of extremely large values since it implies zero union, as
        # per union formula.
        # _iou = _iou * ((_iou >= 0) & (_iou <= (1.0 / EPSILON))).float()
        _iou = _iou * ((_iou >= 0) & (_iou <= LARGE_VALUE)).float()

        return _iou

    def calculate_xy_loss(
            self, prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
        xy_loss = 0
        for b in range(self.B):
            xy_loss += self.λcoord * torch.sum(
                (
                    (self.Iobj_ij * prediction)[
                        ..., self.C + 5 * b: self.C + 5 * b + 2
                    ] - \
                    (self.Iobj_ij * target)[
                        ..., self.C + 5 * b: self.C + 5 * b + 2
                    ]
                ) ** 2,
                dim=None
            )

        return xy_loss

    def calculate_wh_loss(
            self, prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
        """Since predictions can generally produce negative values for w and h,
        their absolute values are taken. Also, small values are added to them
        due to the absolute value function not being differentiable near zero.
        Finally, the square roots are multiplied by sign of the initial
        predictions so that possible negativity is accounted for. None of this
        is required for target values since they are strictly positive."""
        wh_loss = 0
        for b in range(self.B):
            wh_loss += self.λcoord * torch.sum(
                (
                    torch.sign(
                        (self.Iobj_ij * prediction)[
                            ..., self.C + 5 * b + 2:self.C + 5 * b + 4
                        ]
                    ) * \
                    torch.sqrt(
                        torch.abs(
                            (self.Iobj_ij * prediction)[
                                ..., self.C + 5 * b + 2:self.C + 5 * b + 4
                            ] + EPSILON
                        )
                    ) - \
                    torch.sqrt(
                        (self.Iobj_ij * target)[
                            ..., self.C + 5 * b + 2:self.C + 5 * b + 4
                        ]
                    )
                ) ** 2,
                dim=None
            )

        return wh_loss

    def calculate_obj_loss(
            self, prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
        obj_loss = 0
        for b in range(self.B):
            obj_loss += torch.sum(
                ((self.Iobj_ij * prediction)[..., self.C + 5 * b + 4] - \
                (self.Iobj_ij * target)[..., self.C + 5 * b + 4]) ** 2,
                dim=None
            )

        return obj_loss

    def calculate_noobj_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        noobj_loss = 0
        for b in range(self.B):
            noobj_loss += self.λnoobj * torch.sum(
                ((self.Inoobj_ij * prediction)[..., self.C + 5 * b + 4] - \
                torch.zeros_like(
                    (self.Inoobj_ij * prediction)[..., self.C + 5 * b + 4]
                )) ** 2,
                dim=None
            )

        return noobj_loss

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