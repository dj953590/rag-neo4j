import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

"""
    The OCRModel class is a Convolutional Recurrent Neural Network (CRNN) designed for Optical Character Recognition (OCR). 
    Here's a detailed explanation of each part of the class:
    Explanation:
    Imports:  
    torch.nn as nn: Importing the neural network module from PyTorch.
    Class Definition:  
    class OCRModel(nn.Module): Defines the OCRModel class, which inherits from nn.Module.
    Constructor:  
    def __init__(self, num_classes): The constructor initializes the model.
    super(OCRModel, self).__init__(): Calls the constructor of the parent class nn.Module.
    Convolutional Layers:  
    self.cnn = nn.Sequential(...): Defines a sequential container for the convolutional layers.
    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1): First convolutional layer with 1 input channel, 64 output channels, 3x3 kernel, stride of 1, and padding of 1.
    nn.ReLU(): ReLU activation function.
    nn.MaxPool2d(2, 2): Max pooling layer with a 2x2 window.
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1): Second convolutional layer with 64 input channels, 128 output channels, 3x3 kernel, stride of 1, and padding of 1.
    nn.ReLU(): ReLU activation function.
    nn.MaxPool2d(2, 2): Max pooling layer with a 2x2 window.
    Recurrent Layers:  
    self.rnn = nn.LSTM(128, 256, bidirectional=True, batch_first=True): Defines a bidirectional LSTM with 128 input features, 256 hidden units, and batch_first=True.
    Fully Connected Layer:  
    self.fc = nn.Linear(512, num_classes): Defines a fully connected layer with 512 input features (256 from each direction of the bidirectional LSTM) and num_classes output features.
    Forward Method:
    def forward(self, x): Defines the forward pass of the model.
    features = self.cnn(x): Passes the input x through the convolutional layers.
    b, c, h, w = features.size(): Gets the dimensions of the features.
    features = features.squeeze(2).permute(0, 2, 1): Reshapes and permutes the features to match the input format expected by the LSTM.
    recurrent, _ = self.rnn(features): Passes the features through the LSTM.
    output = self.fc(recurrent): Passes the output of the LSTM through the fully connected layer.
    return output: Returns the final output.
"""


class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        # Define the convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # First convolutional layer
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(2, 2),  # Max pooling layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Second convolutional layer
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(2, 2)  # Max pooling layer
        )
        # Define the recurrent layers (LSTM) and fully connected layer
        self.rnn = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        # The fully connected layer outputs the final predictions
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Pass the input through the convolutional and recurrent layers, and return the final predictions
        """
        features = self.cnn(x)
        b, c, h, w = features.size()
        # Reshape the features to have the sequence dimension first
        features = features.squeeze(2).permute(0, 2, 1)
        # Pass the features through the recurrent layers
        recurrent, _ = self.rnn(features)
        # Reshape the output to be compatible with the fully connected layer
        output = self.fc(recurrent)
        return output

    # Define the IAM Dataset class


"""
The IAMDataset class is a custom dataset class for loading and processing 
the IAM dataset, which is commonly used for training 
Optical Character Recognition (OCR) models. 
Here's a detailed explanation of each part of the class:
"""


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


# Load the full IAM dataset
full_dataset = IAMDataset(
    root_dir="./IAM/images",
    labels_file="./IAM/gt_test.txt",
    transform=transform
)

# Split the dataset into training and test sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

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

    for images, labels in train_dataloader:
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

    # Evaluation on test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            input_lengths = torch.full(size=(images.size(0),), fill_value=images.size(3) // 4, dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)

            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)

            loss = criterion(outputs, labels, input_lengths, target_lengths)
            test_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss / len(test_dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "./")

model.eval()  # Set the model to evaluation mode
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
