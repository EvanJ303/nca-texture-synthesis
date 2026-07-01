import torch

def generate_images(targets, model, steps):
    # Create an initial latent state for each image in the batch.
    device = next(model.parameters()).device
    batch_size = targets.size(0)

    states = 0.02 * torch.randn((batch_size, model.state_channels, targets.size(2), targets.size(3)), device=device)

    # Run the cellular update rule for the requested number of steps.
    for _ in range(steps):
        states = model(states)
    
    # Convert the evolved state to RGB values for the final image.
    rgb = torch.tanh(states[:, :3, :, :])
    return rgb