"""
Weight initialization utilities for DualSTAGE.

This module provides standard weight initialization schemes for various
neural network layers used in the DualSTAGE model.
"""

import torch.nn as nn
import math


def init_weights(m):
    """
    Initialize weights for various layer types.
    
    This function applies appropriate initialization schemes based on layer type:
    - Linear layers: Xavier uniform initialization
    - GRU layers: Xavier uniform for input weights, orthogonal for hidden weights
    - Conv1d layers: Kaiming uniform initialization
    - Embedding layers: Normal initialization
    
    Args:
        m: A neural network module (layer)
    """
    if isinstance(m, nn.Linear):
        # Xavier uniform initialization for linear layers
        # Good for layers with tanh/sigmoid activations
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.GRU):
        # Initialize GRU weights
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights: Orthogonal (preserves gradient flow)
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases: Zero initialization
                nn.init.zeros_(param.data)
                # Optionally, initialize forget gate bias to 1 (helps with gradient flow)
                # This is a common practice in LSTM/GRU
                n = param.size(0)
                forget_gate_start = n // 3
                forget_gate_end = 2 * n // 3
                param.data[forget_gate_start:forget_gate_end].fill_(1.0)
    
    elif isinstance(m, nn.LSTM):
        # Initialize LSTM weights (similar to GRU)
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Initialize forget gate bias to 1
                n = param.size(0)
                forget_gate_start = n // 4
                forget_gate_end = n // 2
                param.data[forget_gate_start:forget_gate_end].fill_(1.0)
    
    elif isinstance(m, nn.Conv1d):
        # Kaiming uniform initialization for convolutional layers
        # Good for layers with ReLU activations
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    
    elif isinstance(m, nn.Embedding):
        # Normal initialization for embedding layers
        nn.init.normal_(m.weight, mean=0, std=0.01)
    
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
        # Batch normalization and Layer normalization
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def reset_parameters(model):
    """
    Reset all parameters in a model using init_weights.
    
    This is a convenience function that applies init_weights to all
    modules in the model.
    
    Args:
        model: A PyTorch nn.Module
    """
    model.apply(init_weights)
    return model

