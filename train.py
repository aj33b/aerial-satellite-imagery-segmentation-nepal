import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

from generator import NepalDataset, NepalDataGenerator
from pan import PAN


def train_model(model, train_dataloader, num_epochs=10, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training completed.")


def train_model_with_pan():
    # Step 1: Prepare the dataset as shown in your code

    # Step 2: Create PAN model
    in_channels = 4  # Assuming RGB images
    num_classes = 2  # Number of classes (e.g., background and land cover)
    pan_model = PAN(in_channels)

    # Step 3: Train the PAN model
    data_path = "./output/256x256/"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = NepalDataset(data_path, transform=transform)
    batch_size = 4
    shuffle = True
    data_generator = NepalDataGenerator(dataset, batch_size=batch_size, shuffle=shuffle)
    train_model(pan_model, data_generator)
