import sys
import os

import torch.nn as nn

PROJ_DIR = os.getcwd()
sys.path.insert(0, PROJ_DIR)

from utils.base_model import BaseModel

class SurvivabilityClassifier(BaseModel):
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
    
class SurvivabilityClassifierNN(BaseModel):
    """Class for a neural network classification model
    """
    def __init__(self, n_input: int, threshold=0.5, p_dropout=0.5) -> None:
        """Init a neural network for classification

        Args:
            n_input (int): Dimension of data
            threshold (float, optional): Decision boundary. Defaults to 0.5.
        """
        super().__init__()
                
        self._linear1 = nn.Linear(in_features=n_input, out_features=2*n_input)
        self._linear2 = nn.Linear(in_features=2*n_input, out_features=2*n_input)
        self._linear3 = nn.Linear(in_features=2*n_input, out_features=2*n_input-10)
        self._linear4 = nn.Linear(in_features=2*n_input-10, out_features=1)
        
        self._activation = nn.ReLU()
        self._sigmoid = nn.Sigmoid()
        self._dropout = nn.Dropout(p=p_dropout)
        self._threshold = threshold
        
    def forward(self, X):
        """Run inference

        Args:
            x (Any): Input

        Returns:
            torch.Tensor: Output
        """
        
        X = self._linear1(X)
        X = self._activation(X)
        X = self._dropout(X)
        
        X = self._linear2(X)
        X = self._activation(X)
        X = self._dropout(X)
        
        X = self._linear3(X)
        X = self._activation(X)
        X = self._dropout(X)
        
        X = self._linear4(X)
        X = self._dropout(X)
        
        X =  self._sigmoid(X)
        
        # Convert a tensor of size (nx1) -> a vector of size (n)
        return X.squeeze(1)
    
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
