import torch
import torch.nn as nn

class NCA(nn.Module):
    # Neural Cellular Automaton model with perception and state update.
    def __init__(self, state_channels, hidden_channels, update_prob):
        super().__init__()

        self.state_channels = state_channels
        self.update_prob = update_prob

        # Perception layer extracts local spatial features from state.
        self.perception = nn.Conv2d(
            state_channels,
            hidden_channels, 
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        # Update network computes a state delta from perceived features.
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1, bias=False),
        )

        # Initialize the last layer to produce zero delta initially.
        nn.init.zeros_(self.update[-1].weight)  
    
    def forward(self, state):
        # Compute local perception and update delta for one step.
        features = self.perception(state)
        delta = self.update(features)

        # Random mask drops updates with probability 1 - update_prob.
        prob_mask = (
            torch.rand(
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
                device=state.device
            ) < self.update_prob
        ).float()

        delta = delta * prob_mask

        # Apply the masked delta to the current state.
        return state + delta