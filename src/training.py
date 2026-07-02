import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import models, datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
import matplotlib.pyplot as plt
from model import NCA
from image_generator import generate_images

with open('../config.json', 'r') as f:
    config = json.load(f)

BATCH_SIZE = config['batch_size']
LEARNING_RATE = config['learning_rate']
STATE_CHANNELS = config['state_channels']
HIDDEN_CHANNELS = config['hidden_channels']
UPDATE_PROB = config['update_prob']
MIN_STEPS = config['min_steps']
MAX_STEPS = config['max_steps']
VAL_STEPS = config['val_steps']
NUM_EPOCHS = config['num_epochs']
CLIP_GRAD_NORM = config['clip_grad_norm']
PERCEPTUAL_LOSS_WEIGHT_1 = config['perceptual_loss_weight_1']
PERCEPTUAL_LOSS_WEIGHT_2 = config['perceptual_loss_weight_2']
MSE_LOSS_WEIGHT = config['mse_loss_weight']

def compute_gram_matrix(features):
    # Calculate the Gram matrix for a batch of feature maps.
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, height * width)
    gram_matrix = torch.bmm(features, features.transpose(1, 2))
    gram_matrix /= (channels * height * width)

    return gram_matrix

def calculate_loss(images, targets, slice_1, slice_2, criterion):
    # Move inputs to the model device before computing loss.
    device = next(slice_1.parameters()).device

    images = images.to(device)
    targets = targets.to(device)

    # Compare generated images against the target imagery with pixel-wise loss.
    mse_loss = criterion(images, targets)

    images_01 = (images + 1.0) / 2.0
    targets_01 = (targets + 1.0) / 2.0

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images_norm = normalize(images_01)
    targets_norm = normalize(targets_01)

    # Extract feature maps from intermediate VGG layers for perceptual loss.
    features_images_1 = slice_1(images_norm)
    features_images_2 = slice_2(images_norm)

    with torch.no_grad():
        features_targets_1 = slice_1(targets_norm)
        features_targets_2 = slice_2(targets_norm)

    # Compute Gram matrices for the feature maps to capture style information.
    features_images_1_gram = compute_gram_matrix(features_images_1)
    features_images_2_gram = compute_gram_matrix(features_images_2)

    features_targets_1_gram = compute_gram_matrix(features_targets_1)
    features_targets_2_gram = compute_gram_matrix(features_targets_2)

    # Calculate perceptual loss as the mean squared error between Gram matrices of generated and target features.
    perceptual_loss_1 = criterion(features_images_1_gram, features_targets_1_gram)
    perceptual_loss_2 = criterion(features_images_2_gram, features_targets_2_gram)

    # Combine pixel and perceptual objectives into the training loss.
    total_loss = (
        PERCEPTUAL_LOSS_WEIGHT_1 * perceptual_loss_1 +
        PERCEPTUAL_LOSS_WEIGHT_2 * perceptual_loss_2 +
        MSE_LOSS_WEIGHT * mse_loss
    )

    return total_loss

def save_checkpoint(model, epoch):
    # Persist a model snapshot and record the latest checkpoint name.
    os.makedirs('./data/models', exist_ok=True)

    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'nca_model_epoch_{epoch}_{current_timestamp}.pt'
    path = os.path.join('./data/models', filename)
    torch.save(model.state_dict(), path)

    latest_path = os.path.join('./data', 'latest_checkpoint.txt')
    with open(latest_path, 'w', encoding='utf-8') as f:
        f.write(filename)

    print(f'Saved model checkpoint: {path}')

def plot_loss_curve(losses, session_timestamp):
    # Visualize the validation loss trajectory over training epochs.
    os.makedirs('./data/plots', exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()

    filename = f'session_loss_plot_{session_timestamp}.png'
    plot_path = os.path.join('./data/plots', filename)
    fig.savefig(plot_path)
    plt.close(fig)

def main():
    # Use GPU when available and prepare the training session directory.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the EuroSAT dataset and keep the forest class for training.
    dataset = datasets.EuroSAT(
        root='./data',
        download=True,
        transform=transform
    )

    # Filter the dataset to include only images labeled as 'forest' (class index 4).
    forest_targets = [idx for idx, (_, target) in enumerate(dataset) if target == 4]
    dataset = Subset(dataset, forest_targets)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Prepare data loaders for training and validation batches.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the Neural Cellular Automaton model and VGG16 feature extractor for perceptual loss.
    nca_model = NCA(state_channels=STATE_CHANNELS, hidden_channels=HIDDEN_CHANNELS, update_prob=UPDATE_PROB).to(device)

    vgg_model = models.vgg16(pretrained=True).features.to(device).eval()

    for param in vgg_model.parameters():
        param.requires_grad = False

    slice_1 = nn.Sequential(*list(vgg_model.children())[:9]).to(device)
    slice_2 = nn.Sequential(*list(vgg_model.children())[:16]).to(device)

    optimizer = optim.Adam(nca_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    val_losses = []

    for epoch in range(NUM_EPOCHS):
        # Train for one epoch by generating images and updating the model.
        for targets, _ in train_loader:
            steps = torch.randint(MIN_STEPS, MAX_STEPS + 1, (1,), device=device).item()

            images = generate_images(targets=targets, model=nca_model, steps=steps)
            loss = calculate_loss(images=images, targets=targets, slice_1=slice_1, slice_2=slice_2, criterion=criterion)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(nca_model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

        num_val_batches = len(val_loader)
        val_loss = 0.0

        # Evaluate the model on the validation set without gradient computation.
        for targets, _ in val_loader:
            with torch.no_grad():
                images = generate_images(targets=targets, model=nca_model, steps=VAL_STEPS)
                loss = calculate_loss(images=images, targets=targets, slice_1=slice_1, slice_2=slice_2, criterion=criterion)
            val_loss += loss.item()

        val_loss /= num_val_batches
        val_losses.append(val_loss)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')

        plot_loss_curve(val_losses, session_timestamp)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_checkpoint(nca_model, epoch + 1)

if __name__ == '__main__':
    main()