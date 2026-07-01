import os
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
from generator import generate_images

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
STATE_CHANNELS = 32
HIDDEN_CHANNELS = 128
UPDATE_PROB = 0.5
MIN_STEPS = 32
MAX_STEPS = 64
VAL_STEPS = 64
NUM_EPOCHS = 100
CLIP_GRAD_NORM = 1.0
PERCEPTUAL_LOSS_WEIGHT_1 = 0.3
PERCEPTUAL_LOSS_WEIGHT_2 = 0.7
MSE_LOSS_WEIGHT = 0.1

def calculate_loss(images, targets, slice_1, slice_2, pixel_wise_criterion, perceptual_criterion):
    # Move inputs to the model device before computing loss.
    device = next(slice_1.parameters()).device

    images = images.to(device)
    targets = targets.to(device)

    # Compare generated images against the target imagery with pixel-wise loss.
    mse_loss = pixel_wise_criterion(images, targets)

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

    perceptual_loss_1 = perceptual_criterion(features_images_1, features_targets_1)
    perceptual_loss_2 = perceptual_criterion(features_images_2, features_targets_2)

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

def main():
    # Use GPU when available and prepare the training session directory.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs('./data/plots', exist_ok=True)

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

    forest_targets = [idx for idx, (_, target) in enumerate(dataset) if target == 4]
    dataset = Subset(dataset, forest_targets)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Prepare data loaders for training and validation batches.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    nca_model = NCA(state_channels=STATE_CHANNELS, hidden_channels=HIDDEN_CHANNELS, update_prob=UPDATE_PROB).to(device)

    vgg_model = models.vgg16(pretrained=True).features.to(device).eval()

    for param in vgg_model.parameters():
        param.requires_grad = False

    slice_1 = nn.Sequential(*list(vgg_model.children())[:9]).to(device)
    slice_2 = nn.Sequential(*list(vgg_model.children())[:16]).to(device)

    optimizer = optim.Adam(nca_model.parameters(), lr=LEARNING_RATE)
    pixel_wise_criterion = nn.MSELoss()
    perceptual_criterion = nn.SmoothL1Loss()

    val_losses = []

    for epoch in range(NUM_EPOCHS):
        # Train for one epoch by generating images and updating the model.
        for batch_idx, (targets, _) in enumerate(train_loader):
            steps = torch.randint(MIN_STEPS, MAX_STEPS + 1, (1,), device=device).item()

            images = generate_images(targets=targets, model=nca_model, steps=steps)
            loss = calculate_loss(images=images, targets=targets, slice_1=slice_1, slice_2=slice_2, pixel_wise_criterion=pixel_wise_criterion, perceptual_criterion=perceptual_criterion)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(nca_model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

        num_val_batches = len(val_loader)
        val_loss = 0.0

        for batch_idx, (targets, _) in enumerate(val_loader):
            with torch.no_grad():
                images = generate_images(targets=targets, model=nca_model, steps=VAL_STEPS)
                loss = calculate_loss(images=images, targets=targets, slice_1=slice_1, slice_2=slice_2, pixel_wise_criterion=pixel_wise_criterion, perceptual_criterion=perceptual_criterion)
            val_loss += loss.item()

        val_loss /= num_val_batches
        val_losses.append(val_loss)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')

        fig = plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(val_losses) + 1), val_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss over Epochs')
        plt.grid(True)
        plt.tight_layout()

        filename = f'session_loss_plot_{session_timestamp}.png'
        plot_path = os.path.join('./data/plots', filename)
        fig.savefig(plot_path)
        plt.close(fig)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_checkpoint(nca_model, epoch + 1)

if __name__ == '__main__':
    main()