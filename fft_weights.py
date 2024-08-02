import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import *

# Load the model weights
model = arch1000_1()
model_weights = torch.load('weights/fft1000_1.pt')['state_dict']

# Function to analyze and rank layer contributions
def analyze_weights(model_weights):
    layer_stats = []

    for name, param in model_weights.items():
        if 'weight' in name:
            weights = param.detach().cpu().numpy().flatten()
            variance = np.var(weights)
            layer_stats.append((name, variance))

    layer_stats_dict = {}
    for layer_stat in layer_stats:
        print(layer_stat)
        key = layer_stat[0].split('.')[0]
        if key not in layer_stats_dict:
            layer_stats_dict[key] = []
        layer_stats_dict[key].append(layer_stat[1])

    layer_stats = []
    for key in layer_stats_dict:
        avg_var = np.mean(layer_stats_dict[key])
        layer_stats.append((key, avg_var))

    # Sort layers by variance in descending order
    layer_stats.sort(key=lambda x: x[1], reverse=True)

    return layer_stats

# Analyze the weights
layer_contributions = analyze_weights(model_weights)

# Extract layer names and variances
layer_names = [name for name, _ in layer_contributions]
variances = [variance for _, variance in layer_contributions]

# Plotting
plt.figure(figsize=(12, 8))
plt.barh(layer_names, variances, color='skyblue')
plt.xlabel('Variance')
plt.ylabel('Layer')
plt.title('Layer Variance Ordered by Impact')
plt.gca().invert_yaxis()  # Highest variance on top
plt.tight_layout()
plt.savefig('analysis/figures/fft_weights.png')