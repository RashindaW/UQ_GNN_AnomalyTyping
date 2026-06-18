"""
Configuration module for DualSTAGE (Dynamic Spectral-Temporal Graph Attention).

This module provides a global configuration object that can be accessed
throughout the codebase. It should be initialized with dataset-specific
parameters before model instantiation.
"""

import torch


class DatasetConfig:
    """Dataset-specific configuration."""
    def __init__(self):
        # Default values for refrigeration system (Option A: Medium)
        # Note: Originally 30, but 2 flow sensors had no data and were removed
        self.n_nodes = 28  # Number of sensor nodes (measurement variables)
        self.window_size = 15  # Sliding window size for temporal sequences
        self.ocvar_dim = 6  # Dimension of operating condition variables (control variables)
        self.pred_horizon = 0  # Prediction horizon (0 means reconstruction)
        self.task = "reconstruction"  # reconstruction | prediction
        

class ModelConfig:
    """Model-specific configuration."""
    def __init__(self):
        self.dualstage = DualSTAGEConfig()


class DualSTAGEConfig:
    """DualSTAGE-specific configuration."""
    def __init__(self):
        self.add_self_loop = True  # Whether to add self-loops in feature graph attention


class Config:
    """Global configuration object."""
    def __init__(self):
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def set_dataset_params(self, n_nodes, window_size, ocvar_dim, pred_horizon=0, task="reconstruction"):
        """
        Convenience method to set dataset parameters.
        
        Args:
            n_nodes: Number of sensor nodes
            window_size: Sliding window size
            ocvar_dim: Operating condition variable dimension
            pred_horizon: Prediction horizon length (0 for reconstruction)
            task: Task type ("reconstruction" or "prediction")
        """
        self.dataset.n_nodes = n_nodes
        self.dataset.window_size = window_size
        self.dataset.ocvar_dim = ocvar_dim
        self.dataset.pred_horizon = pred_horizon
        self.dataset.task = task
    
    def validate(self):
        """Validate that all required configuration parameters are set."""
        assert self.dataset.n_nodes is not None, "dataset.n_nodes must be set"
        assert self.dataset.window_size is not None, "dataset.window_size must be set"
        assert self.dataset.ocvar_dim is not None, "dataset.ocvar_dim must be set"
        assert self.dataset.task in ("reconstruction", "prediction"), "dataset.task must be reconstruction or prediction"
        if self.dataset.task == "prediction":
            assert self.dataset.pred_horizon and self.dataset.pred_horizon > 0, "dataset.pred_horizon must be > 0 for prediction"
        return True


# Global configuration instance
cfg = Config()


# Example usage:
# from dualstage.src.config import cfg
# cfg.set_dataset_params(n_nodes=17, window_size=15, ocvar_dim=4)
# cfg.device = 'cuda'
