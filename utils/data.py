from torch.utils.data import TensorDataset
from torch import Tensor

def create_dataset(X, y) -> TensorDataset:
    """Create a dataset from a pair of data and labels

    Args:
        X (torch.Tensor): Data
        y (torch.Tensor): Labels

    Returns:
        TensorDataset: The dataset
    """
    
    X = X.clone().detach()
    y = y.clone().detach()
    
    return TensorDataset(X, y)

def split_data(data,
               train_fraction: float,
               val_fraction: float,
               test_fraction=0.0) -> tuple:
    """Split the dataset into train dataset, validation dataset, and test dataset

    Args:
        dataset: The original dataset
        train_fraction (float): Percentage of the dataset spent for the training set.
        val_fraction (float): Percentage of the dataset spent for the validation set.
        test_fraction (float): Percentage of the dataset spent for the test set. Defaults to 0 (Test set is given as a separated dataset)

    Returns:
        tuple(TensorDataset, TensorDataset, TensorDataset): (train, val, test)
    """
    
    assert train_fraction >= 0
    assert val_fraction >= 0
    assert test_fraction >= 0
    
    train_set = None
    val_set = None
    test_set = None
    
    train_idx = int(len(data) * train_fraction)
    train_set =  data[: train_idx]
    
    val_idx = train_idx + int(len(data) * train_fraction)
    val_set = data[train_idx : val_idx]
    
    if test_fraction > 0:
        test_set = data[val_idx :]
        assert len(train_set) + len(val_set) + len(test_set) == len(data)
    else:
        assert len(train_set) + len(val_set) == len(data)
    
    return (train_set, val_set, test_set)
