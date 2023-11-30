import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('GPU detected: ', torch.cuda.get_device_name())
else:
    DEVICE = torch.device('cpu')
    print('No GPU detected. Using CPU.')

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def __str__(self) -> str:
        return super().__str__()
    
    def predict(X):
        pass

    def train_model(self, 
                    train_dataset: TensorDataset, val_dataset: TensorDataset, 
                    loss_func, optimizer: torch.optim.Optimizer,
                    eval_metric, 
                    epoch_num=1000, batch_size=8, 
                    show_progress=True, show_graph=True):
        """Train the model

        Args:
            train_dataset (TensorDataset): The dataset for training the mode;
            val_dataset (TensorDataset): The dataset for evaluating the model
            loss_func (_type_): Loss function
            optimizer (torch.optim.Optimizer): Optimizer function (Schotastic Gradient Descent, Adam, ...)
            eval_metric (function): The metric used to evaluate the model (F1, accuracy, ...)
            The metric used must follow the following prototye:\n
            def metric(targets: list, predictions: list) -> float\n
            epoch_num (int, optional): Number of iterations. Defaults to 1000.
            batch_size (int, optional): Batch size. Defaults to 8.
            show_progress (bool, optional): Show trainng progress (%). Defaults to True.
            show_graph (bool, optional): Show loss & score vs iteration graph. Defaults to True.
        """
        
        # Set the model in training mode
        self.train()
        torch.cuda.empty_cache()
                
        loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        if show_progress:
            loader = tqdm(loader)
                        
        print(f'\n{"Epoch":>10}  {"Train loss":>15} {"Evaluation score":>15}')
        
        for epoch in range(0, epoch_num):
            total_loss = 0
            
            for (X, y) in loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                
                optimizer.zero_grad()
                
                outputs = self(X)

                loss = loss_func(outputs, y)
                total_loss += float(loss)
                
                loss.backward()
                optimizer.step()
            
            score = self.evaluate(val_dataset, batch_size, eval_metric)
            if epoch == 0 or epoch == epoch_num - 1 or epoch % 10 == 0:
                print(f'{epoch:>10} {total_loss:>25} {score:>25}')
                
    def evaluate(self, dataset, batch_size, metric):
        """Evaluate the perforamane of a model

        Args:
            dataset (TensorDataset): Validation dataset
            batch_size (_type_): Batch size
            metric (function): The metric used to evaluate the model (F1, accuracy, ...)\n
            The metric used must follow the following prototye:
            def metric(targets: list, predictions: list) -> float

        Returns:
            float: Score of the model
        """
        
        self.eval()
        self.cpu()
        
        predictions = []
        labels = []
        
        loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        
        with torch.no_grad():
            for (X, y) in loader:
                predictions.extend(self.predict(X))
                labels.extend(y)

        score = metric(labels, predictions)
                
        self.to(DEVICE)
        self.train()
        
        return score
    
    def save_model(path: str):
        # TODO
        pass
    
    def load_model(path: str):
        # TODO
        pass
                