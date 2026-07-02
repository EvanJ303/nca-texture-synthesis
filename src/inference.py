import os
import json
from datetime import datetime
import torch

with open('../config.json', 'r') as f:
    config = json.load(f)

NUM_STEPS = config['val_steps']