# NCA Texture Synthesis
Neural cellular automata applied to texture generation.

## Overview
The model learns a local update rule that is then applied iteratively to a grid of cells in order to produce an RGB image that matches the style of the target image.

## How it works
-Each cell updates itself using a local rule that depends on the cell's current state as well as those of its neighbors. Only a certain percentage of the cells are allowed to update each step, ensuring that generation is stochastic and asynchronous. 
-The update rule is trained by comparing generated images to a target image using Gram matrix loss, which indicates how well the NCA was able to replicate the structure and style of the target image. 
-The NCA is trained to evolve not just new states but also partially-generated states, forcing the model to learn reliable generation and avoid collapse after a long rollout.

## Requirements
-Python 3.9+
-See requirements.txt for dependencies.

## Usage
-Training parameters, image generation settings, and model architecture can all be adjusted in config.json.
-Run training.py to train the NCA on the target image specified in config.json.
-After training, model checkpoints and loss plots can be accessed in the newly created "data" folder.
-Run inference.py to generate images using the saved NCA model.
-By default, inference.py will use the most recently created model checkpoint, but a different checkpoint can be specified by editing latest_checkpoint.txt.

## Acknowledgements
This project draws inspiration from the methods described in:

Mordvintsev, A., Niklasson, E., & Randazzo, E. (2021). *Texture Generation with 
Neural Cellular Automata*. arXiv:2105.07299. https://arxiv.org/abs/2105.07299

This is an independent implementation for personal learning purposes and is not 
affiliated with or endorsed by the original authors.