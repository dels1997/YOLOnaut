import torch.nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Bottom(torch.nn.Module):
    """Discard the final two fully-connected layers so that we are able to add a
    customized detection head, ensuring that the spatial information and object
    localization ability is retained and then leave the high-level reasoning to
    the YOLO head we add."""
    def __init__(self, original_model: torch.nn.Module) -> None:
        super(ResNet50Bottom, self).__init__()
        self.features = \
            torch.nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x

class YOLOv1(torch.nn.Module):
    def __init__(self, S: int, B: int, C: int) -> None:
        # TODO: Pass argument to decide on pretraining.
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.backbone = ResNet50Bottom(
            original_model=resnet50(weights=ResNet50_Weights.DEFAULT)
        )
        # TODO: Try training with and without freezing the backbone net.
        self.backbone.requires_grad_(False)

        self.pool = torch.nn.Conv2d(2048, 1024, (1, 1))
        self.conv = torch.nn.Sequential(
            # TODO: Parameterize numbers of channels.
            torch.nn.Conv2d(1024, 1024, (3, 3)),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(1024, 1024, (3, 3), stride=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(1024, 1024, (3, 3)),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(1024, 1024, (3, 3)),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 4096),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, self.S * self.S * (self.C + 5 * self.B))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = self.conv(x)
        return x