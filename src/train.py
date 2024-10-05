import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import argparse
import os
# import matplotlib.pyplot as plt
# import seaborn as sns
from utils import plot_confusion_matrix  # Utility for visualization
print('available or not --------->>>>>>>>>>>>>>>>>', torch.cuda.is_available())

def load_config():
    """Load the configuration file."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    return config


def initialize_tensorboard(log_dir):
    """Initialize TensorBoard writer."""
    return SummaryWriter(log_dir=log_dir)


def prepare_data(config):
    """Prepare data loaders for training, validation, and testing."""
    # Data augmentation and transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=config['data']['dataset_path'], transform=transform)
    print(f"Classes:-------->>>>>>>>>>>>>>>>>>>> {dataset.classes}")
    print('traindataset ----->>>>>>>>>>>>>>>', dataset.class_to_idx)  # Should output something like {'cats': 0, 'dogs': 1}
    # Split the dataset
    train_size = int(config['split']['train'] * len(dataset))
    val_size = int(config['split']['val'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
   

    train_loader = DataLoader(train_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader


def initialize_model(config):
    """Load and configure the pre-trained ResNet model."""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, config['model']['num_classes'])
    return model.to(config['device'])


def initialize_optimizer(model, config):
    """Initialize loss function, optimizer, and learning rate scheduler."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler']['step_size'], gamma=config['scheduler']['gamma'])
    
    return criterion, optimizer, scheduler


def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, writer, config):
    """Train the model."""
    best_val_loss = float('inf')  # Initialize the best validation loss to a large number

    for epoch in range(config['hyperparameters']['epochs']):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            print(f"Labels:::::::::::::::::::: {labels}")  # Check the labels
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Log training loss and accuracy for the epoch
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        print(f"Epoch [{epoch + 1}/{config['hyperparameters']['epochs']}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config['device']), labels.to(config['device'])

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(f"Epoch [{epoch + 1}/{config['hyperparameters']['epochs']}], "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['model']['save_dir'], 'best_model.pth'))
            print(f"Best model saved at epoch {epoch + 1}")

        # Step the learning rate scheduler
        scheduler.step()


def evaluate_model(test_loader, model, criterion, config):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate test loss and accuracy
    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Classification report
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))

    # Confusion matrix
    # cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(all_labels, all_preds, classes=config['model']['num_classes'])


def train_the_model():
    config = load_config()
    print(config)
    writer = initialize_tensorboard(config['tensorboard']['log_dir'])

    train_loader, val_loader, test_loader = prepare_data(config)
    model = initialize_model(config)
    criterion, optimizer, scheduler = initialize_optimizer(model, config)

    train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, writer, config)
    evaluate_model(test_loader, model, criterion, config)

    # Close TensorBoard writer
    writer.close()



