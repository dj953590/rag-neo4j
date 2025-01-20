import os
import warnings
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
            nn.MaxPool2d(2, 2),  # Max pooling layer
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Third convolutional layer
            nn.ReLU(),  # Activation function
            nn.MaxPool2d((2, 1)),  # Max pooling layer to reduce height
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Fourth convolutional layer
            nn.ReLU(),  # Activation function
            nn.MaxPool2d((4, 1))  # Max pooling layer to reduce height to 1
        )

        # Define the recurrent layers (LSTM) and fully connected layer
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        # The fully connected layer outputs the final predictions
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Pass the input through the convolutional and recurrent layers, and return the final predictions
        """
        features = self.cnn(x)
        b, c, h, w = features.size()
        # Reshape the features to have the sequence dimension first
        if torch.tensor(h) != torch.tensor(1):
            warnings.warn(f"Expected height after CNN to be 1, but got {h}. Skipping this batch.")
            return None  # Return None to indicate a problematic batch

        features = features.permute(0, 3, 1, 2).contiguous().view(b, w, -1)
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
                # check if image file exists
                file_path = os.path.join(self.root_dir, img_name)
                if os.path.isfile(file_path):
                    self.image_labels.append((img_name, label))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
            # Debug: Check pixel values
        print(f"Image {img_name} - Min: {image.min()}, Max: {image.max()}")
        # Convert label to a sequence of integers (one-hot encoded classes)

        label_tensor = torch.tensor([ord(c) - ord(" ") for c in label], dtype=torch.long)
        return image, label_tensor


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels = torch.cat(labels)
    return images, labels, label_lengths


if __name__ == '__main__':
    # Data preprocessing and augmentation
    transform = transforms.Compose([
        transforms.Resize((32, 128)),  # Resize images to a fixed size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # get the train directory path
    train_dir = os.path.join(os.getcwd(), 'IAM', 'images')
    # get the labels file path
    labels_file = os.path.join(os.getcwd(), 'IAM', 'gt_test.txt')

    # Load the full IAM dataset
    full_dataset = IAMDataset(
        root_dir=train_dir,
        labels_file=labels_file,
        transform=transform
    )

    # Split the dataset into training and test sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4,
                                  collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4,
                                 collate_fn=custom_collate_fn)

    # Initialize the model, loss function, and optimizer
    num_classes = 95  # ASCII characters (32-126) + 1 for blank
    model = OCRModel(num_classes)
    criterion = nn.CTCLoss(blank=0)  # CTC loss for sequence alignment
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels, label_lengths in train_dataloader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Get the input length and target length for CTC
            input_lengths = torch.full(size=(images.size(0),), fill_value=images.size(3) // 4, dtype=torch.long).to(
                device)
            target_lengths = label_lengths.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            if outputs is None:
                continue  # Skip this batch
            # Check for NaN or Inf in outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                warnings.warn("NaN or Inf detected in model outputs. Skipping this batch.")
                continue

            outputs = outputs.permute(1, 0, 2)  # CTC expects (seq_length, batch, num_classes)

            # Compute loss
            try:
                loss = criterion(outputs, labels, input_lengths, target_lengths)
                if torch.isnan(loss) or torch.isinf(loss):
                    warnings.warn("NaN or Inf loss detected. Skipping this batch.")
                    print(f"Outputs: {outputs}")
                    print(f"Labels: {labels}")
                    print(f"Input lengths: {input_lengths}")
                    print(f"Target lengths: {target_lengths}")
                    continue
            except Exception as e:
                warnings.warn(f"Exception in loss computation: {e}. Skipping this batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
            print(f"\rBatch train {len(train_dataloader) * (epoch + 1)} / {len(train_dataloader) * num_epochs}, Loss: {loss.item():.4f}", end="")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

        # Evaluation on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, labels, label_lengths in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                input_lengths = torch.full(size=(images.size(0),), fill_value=images.size(3) // 4, dtype=torch.long).to(
                    device)
                target_lengths = label_lengths.to(device)

                outputs = model(images)
                if outputs is None:
                    continue

                outputs = outputs.permute(1, 0, 2)

                try:
                    loss = criterion(outputs, labels, input_lengths, target_lengths)
                    if torch.isnan(loss) or torch.isinf(loss):
                        warnings.warn("NaN or Inf loss detected. Skipping this batch.")
                        continue
                except Exception as e:
                    warnings.warn(f"Exception in loss computation: {e}. Skipping this batch.")
                    continue

                test_loss += loss.item()
                print(f"\rBatch test {len(test_dataloader) * (epoch + 1)} / {len(test_dataloader) * num_epochs}, Loss: {loss.item():.4f}", end="")

        print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss / len(test_dataloader)}")

    # Save the trained model
    torch.save(model.state_dict(), "ocr_model.pth")

    model.eval()  # Set the model to evaluation mode
    # Export the model to ONNX
    dummy_input = torch.randn(1, 1, 32, 128)  # Example input for ONNX export
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
