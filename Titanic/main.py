import os
import sys

import torch
from torch.nn import BCELoss

from tqdm import tqdm

PROJ_DIR = os.getcwd()
PAR_DIR = os.path.dirname(__file__)

sys.path.insert(0, PROJ_DIR)

from utils.files import DataFile, to_tensor, export_to_csv
from utils.models import BinaryLogisticRegression
from utils.data import create_dataset, split_data

from sklearn.metrics import accuracy_score

def preprocess_datafile(file: DataFile) -> None:
    """Preprocess CSV files. This method will remove redundant columns 
    and fill missing values with the average of the corresponding column (Numerical columns only)

    Args:
        file (DataFile): Input file
    """
    
    # Remove redundant columns
    file.remove_column('Name')
    file.remove_column('Ticket')
    file.remove_column('PassengerId')
    # Because there are too many missing fields (NaN) in this column so it will be removed
    file.remove_column('Cabin')
    
    # Replace missing values (NaN)
    avg_age = file.get_column_avg('Age') # Replace with the average of the column
    file['Age'] = file['Age'].fillna(avg_age)
    
    avg_fare = file.get_column_avg('Fare')
    file['Fare'] = file['Fare'].fillna(avg_fare)
    
    file['Embarked'] = file['Embarked'] .fillna('S') # Embarked: S: 644, C: 168, Q: 77 => S is the most prevailent value
    
    file.apply_hot_encoding('Sex', drop_first=True)
    file.apply_hot_encoding('Embarked')

def tune_hyperparams(regs, momentums, train_set, val_set) -> tuple():
    """Tune the Logistic Regression model's regularization and momentum

    Args:
        regs (list(float)): A list of regularization values
        momentums (list(float)): A list of momentum values
        train_set (torch.TensorDataset): The dataset for training
        val_set (torch.TensorDataset): The dataset for evaluation

    Returns:
        tuple(float, float): A pair of regularization and momentum that produces the highest accuracy
    """
    
    progress = tqdm(total=len(regs) * len(momentums))
    
    best_acc = 0.0
    best_params = (0.0, 0.0)
    
    for reg in regs:
        for momentum in momentums:
            # Init the model
            model = BinaryLogisticRegression(n_input=train_set[0][0].shape[0], threshold=0.5).cuda()
                
            # Train the model    
            model.train_model(train_set, 
                        val_set, 
                        eval_metric=accuracy_score,
                        loss_func=BCELoss(),
                        optimizer=torch.optim.SGD(params=model.parameters(), lr=1e-4, weight_decay=reg, momentum=momentum),
                        batch_size=64,
                        epoch_num=3000,
                        show_progress=False)
            
            model_acc = model.evaluate(val_set, batch_size=32, metric=accuracy_score)
            
            progress.update(1)
            
            if model_acc > best_acc:
                best_acc = model_acc
                best_params = (reg, momentum)
            
            print('Regularization: {r}     Momentum: {m}     Accuracy: {acc}'.format(r = reg, m=momentum, acc=model_acc))
            
    print('The highest accuracy: {acc} was achieved with regularization = {r}, momentum = {m}'.format(acc=best_acc, r=best_params[0], m=best_params[1]))
            
    return best_params


if __name__ == '__main__':    
    train_file = DataFile(os.path.join(PAR_DIR, 'data/train.csv'))
    test_file = DataFile(os.path.join(PAR_DIR, 'data/test.csv'))
    
    preprocess_datafile(train_file)
    preprocess_datafile(test_file)
   
    # Prepare dataset
    X = train_file.get_all_columns_except('Survived')
    y = train_file['Survived']
    
    X_train, X_val, _ = split_data(to_tensor(X), 0.8, 0.2)
    y_train, y_val, _ = split_data(to_tensor(y), 0.8, 0.2)
        
    train_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_val, y_val)
                
    momentums=[0, 0.1, 0.5, 0.9, 0.95, 0.99]
    regs=[1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]
    # best_reg, best_momentum = tune_hyperparams(regs, momentums, train_dataset, val_dataset)
    
    best_reg = 0.01
    best_momentum = 0.95
    
    # Merge the training set and the validation set
    train_dataset = create_dataset(to_tensor(X), to_tensor(y))
    
    model = BinaryLogisticRegression(n_input=train_dataset[0][0].shape[0], threshold=0.5).cuda()
    
    # Train the model    
    model.train_model(train_dataset, 
                val_dataset, 
                eval_metric=accuracy_score,
                loss_func=BCELoss(),
                optimizer=torch.optim.SGD(params=model.parameters(), lr=1e-4, weight_decay=best_reg, momentum=best_momentum),
                batch_size=64,
                epoch_num=3000,
                show_progress=False)
    
    model.cpu()
    X_test = to_tensor(test_file.get_all_columns_except('Survived'))
    predictions = model.predict(X_test)
    
    export_to_csv(predictions, 'PassengerId', 'Survived', start_idx=892)
    