import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Define the OCR Model (CRNN)
class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.squeeze(2).permute(0, 2, 1)
        recurrent, _ = self.rnn(features)
        output = self.fc(recurrent)
        return output

    # Define the IAM Dataset class


class IAMDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        """
        root_dir: Directory containing IAM dataset images.
        labels_file: Path to a file containing image names and labels.
        transform: Optional transformations for the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_labels = []

        # Read the labels file
        with open(labels_file, "r") as f:
            for line in f:
                img_name, label = line.strip().split(maxsplit=1)
                self.image_labels.append((img_name, label))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        # Convert label to a sequence of integers (one-hot encoded classes)
        label_tensor = torch.tensor([ord(c) - ord(" ") for c in label], dtype=torch.long)
        return image, label_tensor


# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((32, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the IAM dataset
dataset = IAMDataset(
    root_dir="path/to/iam/images",
    labels_file="path/to/iam/labels.txt",
    transform=transform
)

# DataLoader for training
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Initialize the model, loss function, and optimizer
num_classes = 95  # ASCII characters (32-126) + 1 for blank
model = OCRModel(num_classes)
criterion = nn.CTCLoss(blank=0)  # CTC loss for sequence alignment
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Get the input length and target length for CTC
        input_lengths = torch.full(size=(images.size(0),), fill_value=images.size(3) // 4, dtype=torch.long).to(device)
        target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.permute(1, 0, 2)  # CTC expects (seq_length, batch, num_classes)

        # Compute loss
        loss = criterion(outputs, labels, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "ocr_model.pth")

# Export the model to ONNX
dummy_input = torch.randn(1, 1, 32, 128).to(device)  # Example input for ONNX export
torch.onnx.export(
    model,
    dummy_input,
    "ocr_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {3: "width"}, "output": {1: "sequence_length"}},
    opset_version=11
)
print("Model exported to onnx.")
