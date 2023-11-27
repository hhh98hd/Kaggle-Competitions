import sys
import os

import torch.nn as nn

from utils.base_model import BaseModel

PROJ_DIR = os.getcwd()
sys.path.insert(0, PROJ_DIR)

class BinaryLogisticRegression(BaseModel):
    """Class for a binary logistic regression model
    """
    def __init__(self, n_input: int, threshold=0.5) -> None:
        """Init a binary logistic regression model

        Args:
            n_input (int): Dimension of data
            threshold (float, optional): Decision boundary. Defaults to 0.5.
        """
        super().__init__()
        
        self._linear = nn.Linear(in_features=n_input, out_features=1)
        self._sigmoid = nn.Sigmoid()
        self._threshold = threshold
        
    def forward(self, x):
        """Run inference

        Args:
            x (Any): Input

        Returns:
            torch.Tensor: Output
        """
        
        x = self._linear(x)
        x =  self._sigmoid(x)
        
        # Convert a tensor of size (nx1) -> a vector of size (n)
        return x.squeeze(1)
    
    def predict(self, X):
        """Make predictions based on the model's outputs

        Args:
            X (Any): The model's outputs

        Returns:
            list(int): Predictions
        """
        predictions = self(X)
                
        predictions[predictions >= self._threshold] = 1
        predictions[predictions < self._threshold] = 0
                
        return predictions
    