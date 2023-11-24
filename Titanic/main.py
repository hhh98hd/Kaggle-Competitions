import os
import sys
import math

from torch.utils.data import TensorDataset
import torch
from torch.nn import BCELoss
import numpy as np

PROJ_DIR = os.getcwd()
PAR_DIR = os.path.dirname(__file__)

sys.path.insert(0, PROJ_DIR)

from utils.files import DataFile, to_tensor
from utils.models import LogisticRegression
from utils.data import create_dataset, split_data

from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # TODO: Check if performing one-hot encoding on Pclass improves the model's performance
    
    train_file = DataFile(os.path.join(PAR_DIR, 'data/train.csv'))
                    
    # Remove redundant columns
    train_file.remove_column('Name')
    train_file.remove_column('Ticket')
    train_file.remove_column('PassengerId')
    train_file.apply_hot_encoding('Sex', drop_first=True)
    train_file.apply_hot_encoding('Embarked')
    # Because there are too many missing fields (NaN) in this column so it will be removed
    train_file.remove_column('Cabin')
        
    # Replace missing values (NaN) with average of the column
    avg_age = train_file.get_column_avg('Age')
    pax_ages = train_file['Age']
    for i in range(0, len(pax_ages)):
        if(math.isnan(pax_ages[i])): pax_ages[i] = avg_age
    train_file['Age'] = pax_ages
        
    # Prepare dataset
    X = train_file.get_all_columns_except('Survived')
    y = train_file['Survived']
    
    X_train, X_val, _ = split_data(to_tensor(X), 0.7, 0.3)
    y_train, y_val, _ = split_data(to_tensor(y), 0.7, 0.3)
        
    train_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_val, y_val)
        
    # Init the model
    model = LogisticRegression(X.shape[1], 1e-3)
    
    # # Train the model    
    # model.train(train_dataset, 
    #             val_dataset, 
    #             eval_metric=accuracy_score,
    #             loss_func=BCELoss(),
    #             optimizer=torch.optim.SGD(params=model.parameters(), lr=1e-4),
    #             epoch_num=3000)

    