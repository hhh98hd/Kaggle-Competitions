from torch.utils.data import TensorDataset
from torch import Tensor

def create_dataset(X, y) -> TensorDataset:
    """Create a dataset from a pair of data and labels

    Args:
        X (_type_): Data
        y (_type_): Labels

    Returns:
        TensorDataset: The dataset
    """
    
    if type(X) != TensorDataset:
        X = X.clone().detach()
    if type(y) != TensorDataset:
        y = y.clone().detach()
        
    return TensorDataset(X, y)

def split_data(data,
               train_portion: float,
               val_portion: float,
               test_portion=0.0) -> tuple:
    """Split the dataset into train dataset, validation dataset, and test dataset

    Args:
        dataset: The original dataset
        train_portion (float): Percentage of the dataset spent for the training set.
        val_portion (float): Percentage of the dataset spent for the validation set.
        test_portion (float): Percentage of the dataset spent for the test set. Defaults to 0 (Test set is given as a separated dataset)

    Returns:
        tuple(TensorDataset, TensorDataset, TensorDataset): (train, val, test)
    """
    
    assert train_portion >= 0
    assert val_portion >= 0
    assert test_portion >= 0
    
    train_set = None
    val_set = None
    test_set = None
    
    train_idx = int(len(data) * train_portion)
    train_set =  data[: train_idx]
    
    val_idx = train_idx + int(len(data) * train_portion)
    val_set = data[train_idx : val_idx]
    
    if test_portion > 0:
        test_set = data[val_idx :]
        assert len(train_set) + len(val_set) + len(test_set) == len(data)
    else:
        assert len(train_set) + len(val_set) == len(data)
    
    return (train_set, val_set, test_set)
