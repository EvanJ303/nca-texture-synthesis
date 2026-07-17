import os
import json
import requests
from datetime import datetime
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.transforms import ToTensor, Normalize
import matplotlib.pyplot as plt
from PIL import Image

from model import NCA
from image_generator import generate_images

with open('./config.json', 'r') as f:
    config = json.load(f)

TARGET_URL = config['training']['target_url']
IMAGE_HEIGHT = config['training']['image_height']
IMAGE_WIDTH = config['training']['image_width']
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = config['training']['learning_rate']
STATE_CHANNELS = config['model']['state_channels']
HIDDEN_CHANNELS = config['model']['hidden_channels']
UPDATE_PROB = config['model']['update_prob']
MIN_STEPS = config['training']['min_steps']
MAX_STEPS = config['training']['max_steps']
NUM_ITERATIONS = config['training']['num_iterations']
CLIP_GRAD_NORM = config['training']['clip_grad_norm']
POOL_SIZE = config['training']['pool_size']
PERCEPTUAL_LOSS_WEIGHT_1 = config['training']['perceptual_loss_weight_1']
PERCEPTUAL_LOSS_WEIGHT_2 = config['training']['perceptual_loss_weight_2']
OVERFLOW_LOSS_WEIGHT = config['training']['overflow_loss_weight']

def get_target_image(url, image_height, image_width, device):
    # Download the target image from the specified URL and preprocess it for training.
    if url.startswith(('http://', 'https://')):
        response = requests.get(url)
        f = BytesIO(response.content)
    else:
        f = url

    target_image = Image.open(f).convert('RGB')
    target_image = target_image.resize((image_width, image_height))

    target_tensor = ToTensor()(target_image).unsqueeze(0).to(device)
    
    return target_tensor

def compute_gram_matrix(features):
    # Calculate the Gram matrix for a batch of feature maps.
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, height * width)
    gram_matrix = torch.bmm(features, features.transpose(1, 2))
    gram_matrix /= (channels * height * width)

    return gram_matrix

def precompute_target_gram_matrix(target, vgg_slice_1, vgg_slice_2):
    # Precompute the Gram matrices for the target image's feature maps to optimize training.
    with torch.no_grad():
        device = next(vgg_slice_1.parameters()).device
        target = target.to(device)

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        target_norm = normalize(target)

        features_target_1 = vgg_slice_1(target_norm)
        features_target_2 = vgg_slice_2(target_norm)

        features_target_1_gram = compute_gram_matrix(features_target_1)
        features_target_2_gram = compute_gram_matrix(features_target_2)

    return features_target_1_gram, features_target_2_gram

def calculate_loss(images, features_target_1_gram, features_target_2_gram, overflow_loss, vgg_slice_1, vgg_slice_2, criterion):
    # Move inputs to the model device before computing loss.
    device = next(vgg_slice_1.parameters()).device

    images = images.to(device)
    features_target_1_gram = features_target_1_gram.to(device)
    features_target_2_gram = features_target_2_gram.to(device)
    overflow_loss = overflow_loss.to(device)

    # Scale images from [-1, 1] to [0, 1] for VGG normalization.
    images = (images + 1.0) / 2.0

    features_target_1_gram = features_target_1_gram.expand(images.size(0), -1, -1)
    features_target_2_gram = features_target_2_gram.expand(images.size(0), -1, -1)

    # Normalize images for VGG feature extraction.
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images_norm = normalize(images)

    # Extract feature maps from intermediate VGG layers for perceptual loss.
    features_images_1 = vgg_slice_1(images_norm)
    features_images_2 = vgg_slice_2(images_norm)

    # Compute Gram matrices for the feature maps to capture style information.
    features_images_1_gram = compute_gram_matrix(features_images_1)
    features_images_2_gram = compute_gram_matrix(features_images_2)

    # Calculate perceptual loss as the mean squared error between Gram matrices of generated and target features.
    perceptual_loss_1 = criterion(features_images_1_gram, features_target_1_gram)
    perceptual_loss_2 = criterion(features_images_2_gram, features_target_2_gram)

    # Combine perceptual objectives into the training loss.
    total_loss = (
        PERCEPTUAL_LOSS_WEIGHT_1 * perceptual_loss_1 +
        PERCEPTUAL_LOSS_WEIGHT_2 * perceptual_loss_2 +
        OVERFLOW_LOSS_WEIGHT * overflow_loss
    )

    return total_loss

def save_checkpoint(model, iteration):
    # Persist a model snapshot and record the latest checkpoint name.
    os.makedirs('./data/models', exist_ok=True)

    filename = f'nca_model_iteration_{iteration}.pt'
    path = os.path.join('./data/models', filename)
    torch.save(model.state_dict(), path)

    latest_path = os.path.join('./data', 'latest_checkpoint.txt')
    with open(latest_path, 'w', encoding='utf-8') as f:
        f.write(filename)

    print(f'Saved model checkpoint: {path}')

def plot_loss_curve(losses, session_timestamp):
    # Visualize the validation loss trajectory over training steps.
    os.makedirs('./data/plots', exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Steps')
    plt.grid(True)
    plt.tight_layout()

    # Save the loss plot with a timestamped filename for session tracking.
    filename = f'session_loss_plot_{session_timestamp}.png'
    plot_path = os.path.join('./data/plots', filename)
    fig.savefig(plot_path)
    plt.close(fig)

def main():
    # Use GPU when available and prepare the training session directory.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize the Neural Cellular Automaton model and VGG16 feature extractor for perceptual loss.
    nca_model = NCA(state_channels=STATE_CHANNELS, hidden_channels=HIDDEN_CHANNELS, update_prob=UPDATE_PROB).to(device)

    vgg_model = models.vgg16(pretrained=True).features.to(device).eval()

    for param in vgg_model.parameters():
        param.requires_grad = False

    # Create slices of the VGG model to extract features from specific layers for perceptual loss computation.
    vgg_slice_1 = nn.Sequential(*list(vgg_model.children())[:9]).to(device)
    vgg_slice_2 = nn.Sequential(*list(vgg_model.children())[:16]).to(device)

    optimizer = optim.Adam(nca_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    target_image = get_target_image(TARGET_URL, IMAGE_HEIGHT, IMAGE_WIDTH, device)
    features_target_1_gram, features_target_2_gram = precompute_target_gram_matrix(target_image, vgg_slice_1, vgg_slice_2)

    # Initialize a pool of states for training, which will be updated iteratively.
    pool = torch.zeros((POOL_SIZE, STATE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=device)

    losses = []

    for iteration in range(NUM_ITERATIONS):
        # Randomly sample a batch of states from the pool for training.
        idx = torch.randperm(POOL_SIZE, device=device)[:BATCH_SIZE]
        states = pool[idx].clone()
        
        if iteration % 8 == 0:
            states[0] = torch.zeros((STATE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=device)

        num_steps = torch.randint(MIN_STEPS, MAX_STEPS + 1, (1,)).item()
        images, overflow_loss = generate_images(states=states, model=nca_model, num_steps=num_steps)

        # Compute the total loss, including perceptual and overflow components, and perform backpropagation to update model parameters.
        loss = calculate_loss(
            images=images,
            features_target_1_gram=features_target_1_gram,
            features_target_2_gram=features_target_2_gram,
            overflow_loss=overflow_loss,
            vgg_slice_1=vgg_slice_1,
            vgg_slice_2=vgg_slice_2,
            criterion=criterion
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(nca_model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        # Add the evolved states back to the pool and update the loss.
        pool[idx] = states.detach()

        losses.append(loss.item())

        if (iteration + 1) % 100 == 0 or (iteration + 1) == NUM_ITERATIONS:
            print(f'Iteration [{iteration + 1}/{NUM_ITERATIONS}], Loss: {loss.item():.4f}, Overflow Loss: {overflow_loss.item():.4f}, Num Steps: {num_steps}')

        # Periodically save model checkpoints and plot the loss curve for monitoring training progress.
        if (iteration + 1) % 1000 == 0 or (iteration + 1) == NUM_ITERATIONS:
            save_checkpoint(nca_model, iteration + 1)
            plot_loss_curve(losses, session_timestamp)

if __name__ == '__main__':
    main()