import os
import json
from datetime import datetime
import torch
from torchvision.transforms.functional import to_pil_image
from model import NCA
from image_generator import generate_images

with open('./config.json', 'r') as f:
    config = json.load(f)

STATE_CHANNELS = config['model']['state_channels']
HIDDEN_CHANNELS = config['model']['hidden_channels']
UPDATE_PROB = config['model']['update_prob']
NUM_IMAGES = config['inference']['num_images']
IMAGE_HEIGHT = config['inference']['image_height']
IMAGE_WIDTH = config['inference']['image_width']
NUM_STEPS = config['inference']['num_steps']

def load_latest_checkpoint(model):
    device = next(model.parameters()).device

    # Load the most recent model checkpoint
    with open('./data/latest_checkpoint.txt', 'r') as f:
        filename = f.read().strip()
    
    path = os.path.join('./data/models', filename)
    
    model.load_state_dict(torch.load(path, map_location=device))

def save_images(images, session_timestamp):
    os.makedirs(f'./data/generated_images/{session_timestamp}', exist_ok=True)

    for idx, image in enumerate(images):
        # Convert the tensor to a PIL image and save it.
        pil_image = to_pil_image((image + 1.0) / 2.0)  # Scale from [-1, 1] to [0, 1]
        pil_image.save(f'./data/generated_images/{session_timestamp}/image_{idx}.png')

def main():
    # Create a timestamp for the current session
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load the model
    nca_model = NCA(state_channels=STATE_CHANNELS, hidden_channels=HIDDEN_CHANNELS, update_prob=UPDATE_PROB)
    load_latest_checkpoint(nca_model)
    nca_model.eval()

    states = torch.zeros((NUM_IMAGES, STATE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))

    # Generate images using the loaded model
    with torch.no_grad():
        images, _ = generate_images(states=states, model=nca_model, num_steps=NUM_STEPS)
    
    # Save the generated images to disk
    save_images(images, session_timestamp)

if __name__ == '__main__':
    main()