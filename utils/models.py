import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        if torch.cuda.is_available():
            self.gpu_available = True
            self.device = torch.device('cuda')
            print('GPU detected: ', torch.cuda.get_device_name())
        else:
            self.gpu_available = False
            self.device = torch.device('cpu')
            print('No GPU detected. Using CPU.')
        
    def __str__(self) -> str:
        return super().__str__()
    
    def predict(X):
        pass

    def train_model(self, train_dataset, val_dataset, loss_func, optimizer: torch.optim.Optimizer, eval_metric, epoch_num=1000, batch_size=8, show_progress=True):
        # Set the model in training mode
        self.train()
        torch.cuda.empty_cache()
                
        loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        if show_progress:
            loader = tqdm(loader)
        
        for epoch in range(0, epoch_num):
            for (X, y) in loader:
                optimizer.zero_grad()
                
                logits = self(X)
                loss = loss_func(logits, y)
                
                loss.backward()
                optimizer.step()
            
            self.evaluate(val_dataset, batch_size, eval_metric)
                
    def evaluate(self, dataset, batch_size, metric):
        self.eval()
        
        predictions = []
        labels = []
        
        loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        
        with torch.no_grad():
            for (X, y) in loader:
                predictions.extend(self.predict(X))
                labels.extend(y)

        score = metric(labels, predictions)
                
        self.train()
        
        return score
                
        
class LogisticRegression(BaseModel):
    """Class for a binary logistic regression model
    """
    def __init__(self, n_input: int, threshold=0.5) -> None:
        """Init a binary logistic regression model

        Args:
            n_input (int): Dimension of data
            threshold (float, optional): Decision boundary. Defaults to 0.5.
        """
        
        super().__init__()
        
        self._liear = nn.Linear(in_features=n_input, out_features=1)
        self._sigmoid = nn.Sigmoid()
        self._threshold = threshold
        
    def forward(self, x):
        x = self._liear(x)
        x =  self._sigmoid(x)
        
        # Convert a tensor of size (nx1) -> a vector of size (n)
        return x.squeeze(1)
    
    def predict(self, X):
        self.eval()
        
        predictions = self(X)
                
        predictions[predictions >= self._threshold] = 1
        predictions[predictions < self._threshold] = 0
                
        return predictions
    