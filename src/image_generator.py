import torch

def generate_images(states, model, num_steps):
    device = next(model.parameters()).device
    states = states.to(device)

    # Run the cellular update rule for the requested number of steps.
    for _ in range(num_steps):
        states = model(states)
    
    overflow_loss = (states - states.clamp(-1.0, 1.0)).abs().sum()

    # Convert the evolved state to RGB values for the final image.
    rgb = torch.tanh(states[:, :3, :, :])
    return rgb, overflow_loss