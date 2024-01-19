from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
import torch

import config
from loss import YOLOLoss
from network import YOLO
from utils.data_utils import CustomDataset

image_directory, target_directory = \
    config.training_parameters["image_directory"], \
    config.training_parameters["target_directory"]

image_width, image_height = \
    config.model_parameters["image_width"], \
    config.model_parameters["image_height"]

version = config.model_parameters["version"]
S, B, C = config.model_parameters["S"], \
          config.model_parameters["B"], \
          config.model_parameters["C"]

batch_size = config.training_parameters["batch_size"]
learning_rate = config.training_parameters["learning_rate"]
num_epochs = config.training_parameters["num_epochs"]
best_model_checkpoint_path = \
    config.training_parameters["best_model_checkpoint_path"]
transform = Compose([Resize((image_width, image_height)), ToTensor()])
dataset = CustomDataset(
    image_directory=image_directory, target_directory=target_directory,
    transform=transform
)

# Include drop_last=True during full training so that there are no issues with
# padding and shapes, and that the memory is used efficiently.
dataloader = \
    DataLoader(dataset, batch_size=batch_size, shuffle=True,
               # drop_last=True
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(version, S=S, B=B, C=C)
model.to(device)

criterion = YOLOLoss(
    version, image_width=image_width, image_height=image_height, S=S, B=B, C=C
)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model = model.state_dict()

        # Save the best model.
        torch.save(best_model, best_model_checkpoint_path)
